#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, build_rotation
from utils.jagged_tensor import JaggedTensor
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH, SH2RGB
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.visualisation_utils import gaussian_kernel_2d, fold_images
from diff_gaussian_rasterization_texture._C import kmeans_cuda
import numpy as np
from collections import OrderedDict
from error_stats import ErrorStats
from collections.abc import Callable

class Codebook():
    def __init__(self, ids, centers):
        self.ids = ids
        self.centers = centers
    
    def evaluate(self):
        return self.centers[self.ids.flatten().long()]

def generate_codebook(values: torch.Tensor, inverse_activation_fn: Callable[[torch.Tensor], torch.Tensor] = lambda x: x, num_clusters: int = 256, tol: float = 0.0001):
    shape = values.shape
    values = values.flatten().view(-1, 1)
    centers = values[torch.randint(values.shape[0], (num_clusters, 1), device="cuda").squeeze()].view(-1,1)

    ids, centers = kmeans_cuda(values, centers.squeeze(), tol, 500)
    ids = ids.byte().squeeze().view(shape)
    centers = centers.view(-1,1)

    return Codebook(ids.cuda(), inverse_activation_fn(centers.cuda()))

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        
        self.texture_map_activation = lambda x: 2*torch.sigmoid(x)-1
        self.inverse_texture_map_activation = lambda x: -torch.log((1 - x) / (x + 1))
        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int, max_texture_resolution: int, texture_cutoff: float = 3):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.percent_dense = 0
        self._codebook_dict = None
        self._max_texture_resolution = max_texture_resolution
        self._texture_map: JaggedTensor = JaggedTensor()
        self._texel_size = None
        self._texture_cutoff = texture_cutoff
        self.error_stats = ErrorStats(0)
        

        self._texel_pixel_ratio = torch.empty(0)
        self._pixel_size = torch.empty(0)
        self.setup_functions()

    def initialise_primitives(self, **kwargs):
        if kwargs['init_type'] == "pcd":
            self.create_from_pcd(kwargs['scene_info'].point_cloud, kwargs['samples_num'])
        elif kwargs['init_type'] == "ply":
            self.load_ply(kwargs["path"])
        self.error_stats = ErrorStats(self.num_primitives)

    @property
    def texel_size(self):
        if self._texel_size is None:
            return self._pixel_size * 2**self._texel_pixel_ratio
        else:
            return self._texel_size

    @property
    def num_primitives(self) -> int:
        return self._xyz.shape[0]

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        # if self.variable_sh_bands:
        #     features = list()
        #     index_start = 0
        #     for idx, sh_tensor in enumerate(self._features_rest):
        #         index_end = index_start + self.per_band_count[idx]
        #         features.append(torch.cat((self._features_dc[index_start: index_end], sh_tensor), dim=1))
        #         index_start = index_end
        # else:
        features = torch.cat((self._features_dc, self._features_rest), dim=1)
        return features

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_texture_map(self):
        return self.texture_map_activation(self._texture_map._values)
    
    @property
    def base_color(self):
        return SH2RGB(self._features_dc[:, 0])
    
    # Alias for get_texture_map
    @property
    def texture_offsets(self):
        return self.texture_map_activation(self._texture_map._values)

    # Broken as no activation is used
    @property
    def color(self):
        raise NotImplementedError()
        return self._texture_map.add(self.base_color.view(-1, 3))
    
    # Conservative way to determine texture resolution
    # The computation is based on the fact that colours with an alpha value
    # of less than 1/255 are not rendered
    def _calculate_active_texture_resolution(self, square: bool = False, powers_of_two: bool = False, return_differentiable: bool = False, clamp_max:bool = True):
        # Find the primitive's extent in canonical space, that is, how much is the extent
        # in standard deviation units
        canonical_extent = 2 * torch.sqrt(2 * torch.log(255*torch.clamp_min(self.get_opacity.detach(), 1/255)))
        canonical_extent.clamp_max_(2 * self._texture_cutoff)

        # Multiply by scaling to get the extent in world units
        world_extent = canonical_extent * self.get_scaling[..., :2]

        # Find how many texels comprises the texture
        if return_differentiable:
            return world_extent / self.texel_size
        texture_extent = torch.ceil(world_extent / self.texel_size).long()

        max_tex_res = torch.tensor((self._max_texture_resolution, self._max_texture_resolution), device="cuda", dtype=torch.int32)

        start = torch.floor((max_tex_res - 1) / 2 - texture_extent/2).clamp_min(0)
        end = torch.ceil((max_tex_res - 1) / 2 + texture_extent/2) + 1

        texture_resolution = (end - start)
        if square:
            texture_resolution = texture_resolution.max(dim=1, keepdim=True).values.repeat((1, 2)).long()
        if powers_of_two:
            texture_resolution = 2**torch.ceil(torch.log2(texture_resolution)).long()
        if clamp_max:
            texture_resolution = texture_resolution.clamp_max(max_tex_res)

        return texture_resolution.clamp_min(1).long()

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, num_samples=-1):
        mask = torch.ones((pcd.points.shape[0]), dtype=torch.bool, device="cuda")
        if num_samples > -1:
            mask = ~mask
            indices = torch.randperm(mask.shape[0])[:num_samples]
            mask[indices] = True
        fused_point_cloud = torch.from_numpy(np.asarray(pcd.points)).float().cuda()[mask]
        fused_color = RGB2SH(torch.from_numpy(np.asarray(pcd.colors)).float().cuda()[mask])

        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 0.0000001)
        dist2 = torch.sqrt(dist2)
        scales = self.scaling_inverse_activation(dist2)[...,None].repeat(1, 3)
        scales[:, 2] = scales[:, 2] - 5
        
        rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = fused_point_cloud.requires_grad_(True)
        self._features_dc = features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True)
        self._features_rest = features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True)
        self._scaling = scales.requires_grad_(True)
        self._rotation = rots.requires_grad_(True)
        self._opacity = opacities.requires_grad_(True)
        
        self._texture_map = JaggedTensor(
            torch.ones((self.num_primitives, 2), device="cuda", dtype=torch.int32) * 2,
            self.inverse_texture_map_activation(0.000001 * torch.ones([self.num_primitives, 2 * 2, 3], device="cuda")).reshape(-1, 3))
        self._texture_map._values.requires_grad_(True)
        self._texel_pixel_ratio = 1 * torch.ones(self.num_primitives, 1, device="cuda", dtype=torch.int32)
    
    def construct_list_of_attributes(self, rest_coeffs=45):
        return ['x', 'y', 'z',
                'f_dc_0','f_dc_1','f_dc_2',
                *[f"f_rest_{i}" for i in range(rest_coeffs)],
                'opacity',
                'scale_0','scale_1',
                'rot_0','rot_1','rot_2','rot_3',
                "texel_size"]

    # TODO fixed jagged array save
    def save_ply(self, path: str):

        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling[:, :2].detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        texel_size = self.texel_size.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, f_dc, f_rest, opacities, scale, rotation, texel_size), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    # TODO Have a optimizer-less version (GaussianModel self contained)
    def save_texture_maps(self, path: str, optimizer = None, quantize: bool = False):
        
        # Store a texture as tight to the current size as possible
        if optimizer:
            self.texture_map_resize(optimizer, powers_of_two=False)
        texture_map = self.get_texture_map.detach()

        if quantize:
            texture_map_codebook = generate_codebook(texture_map, self.inverse_texture_map_activation)
            torch.save(texture_map_codebook, os.path.join(path, "texture_map_codebook.pt"))
        else:
            torch.save(texture_map.cpu(), os.path.join(path, "texture_map.pt"))

    # TODO Ugly, fix it
    def quantize_texture_maps(self, codebook: str | Codebook):
        if isinstance(codebook, str):
            codebook = torch.load(os.path.join(codebook))
        if isinstance(codebook, Codebook):
            self._texture_map._values = codebook.evaluate().view(-1, 3)

    def _parse_vertex_group(self,
                            vertex_group,
                            sh_degree,
                            float_type,
                            attribute_type,
                            max_coeffs_num):
        coeffs_num = (sh_degree+1)**2 - 1
        num_primitives = vertex_group.count

        xyz = np.stack((np.asarray(vertex_group["x"], dtype=float_type),
                        np.asarray(vertex_group["y"], dtype=float_type),
                        np.asarray(vertex_group["z"], dtype=float_type)), axis=1)

        opacity = np.asarray(vertex_group["opacity"], dtype=attribute_type)[..., np.newaxis]
    
        # Stacks the separate components of a vector attribute into a joint numpy array
        # Defined just to avoid visual clutter
        def stack_vector_attribute(name, count):
            return np.stack([np.asarray(vertex_group[f"{name}_{i}"], dtype=attribute_type)
                             for i in range(count)], axis=1)

        features_dc = stack_vector_attribute("f_dc", 3).reshape(-1, 1, 3)
        scaling = stack_vector_attribute("scale", 2)
        rotation = stack_vector_attribute("rot", 4)
        texel_size = np.asarray(vertex_group["texel_size"], dtype=attribute_type)[..., np.newaxis]
        features_rest = stack_vector_attribute("f_rest", coeffs_num*3).reshape((num_primitives, 3, coeffs_num))
        # Using full tensors (P x 15) even for points that don't require it
        features_rest = np.concatenate(
            (features_rest,
                np.zeros((num_primitives, 3, max_coeffs_num - coeffs_num), dtype=attribute_type)), axis=2)


        xyz = torch.from_numpy(xyz).cuda()
        features_dc = torch.from_numpy(features_dc).contiguous().cuda()
        features_rest = torch.from_numpy(features_rest).contiguous().cuda()
        opacity = torch.from_numpy(opacity).cuda()
        scaling = torch.from_numpy(scaling).cuda()
        rotation = torch.from_numpy(rotation).cuda()
        texel_size = torch.from_numpy(texel_size).cuda()

        return {'xyz': xyz,
                'opacity': opacity,
                'features_dc': features_dc,
                'features_rest': features_rest if sh_degree > 0 else None,
                'scaling': scaling,
                'rotation': rotation,
                'texel_size': texel_size,
        }

    def load_ply(self, path):
        plydata = PlyData.read(path)

        max_coeffs_num = (self.max_sh_degree+1)**2 - 1

        attributes_dict = self._parse_vertex_group(plydata.elements[0],
                                                    self.max_sh_degree,
                                                    'f4',
                                                    'f4',
                                                    max_coeffs_num)

        xyz = attributes_dict['xyz']
        features_dc = attributes_dict['features_dc']
        features_rest = attributes_dict['features_rest'].transpose(1,2)
        opacity = attributes_dict['opacity']
        scaling = attributes_dict['scaling']
        rotation = attributes_dict['rotation']
        texel_size = attributes_dict['texel_size']
        
        self._xyz = xyz.requires_grad_(True)
        self._features_dc = features_dc.requires_grad_(True)
        self._features_rest = features_rest.requires_grad_(True)
        self._opacity = opacity.requires_grad_(True)
        self._scaling = torch.cat((scaling, -5 * torch.ones_like(opacity)), dim=-1).requires_grad_(True)
        self._rotation = rotation.requires_grad_(True)
        self._texel_size = texel_size

        self.active_sh_degree = self.max_sh_degree

    def load_texture_maps(self, path: str, quantize: bool = False):
        if quantize:
            texture_map_codebook: Codebook = torch.load(os.path.join(path, "texture_map_codebook.pt"))
            texture_map = texture_map_codebook.evaluate().view(-1, 3)
            
        else:
            texture_map = torch.load(os.path.join(path, "texture_map.pt"))

        # Unactivate texture map values
        texture_map = self.inverse_texture_map_activation(texture_map)
        self._texture_map = JaggedTensor(self._calculate_active_texture_resolution(powers_of_two=False).int(), texture_map.cuda())

    # TODO Have a optimizer-less version (GaussianModel self contained)
    def prune_points(self, mask, optimizer):
        valid_points_mask = ~mask
        optimizable_tensors = optimizer._prune_optimizer(valid_points_mask, self._texture_map._sizes)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._texture_map._values = optimizable_tensors["texture_map"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        # Prune the error stats tensors
        self.error_stats.errors = self.error_stats.errors[valid_points_mask]
        self.error_stats.areas = self.error_stats.areas[valid_points_mask]
        self.error_stats.contributions = self.error_stats.contributions[valid_points_mask]

        self._texel_pixel_ratio = self._texel_pixel_ratio[valid_points_mask]
        self._texture_map._sizes = self._texture_map._sizes[valid_points_mask]

    # TODO Have a optimizer-less version (GaussianModel self contained)
    def densification_postfix(self,
                              optimizer,
                              new_xyz,
                              new_features_dc,
                              new_features_rest,
                              new_texture_map,
                              new_texture_resolution,
                              new_opacities,
                              new_scaling,
                              new_rotation,
                              densification_mask,
                              repeats = 2):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "texture_map": new_texture_map,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = optimizer.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._texture_map._values = optimizable_tensors["texture_map"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._texel_pixel_ratio = torch.cat((self._texel_pixel_ratio, self._texel_pixel_ratio[densification_mask].repeat(repeats, 1)), dim=0)
        self._texture_map._sizes = torch.cat((self._texture_map._sizes, new_texture_resolution), dim=0)


        # self.error_stats.errors = torch.cat((self.error_stats.errors, torch.zeros((densification_mask.sum(), 1), device="cuda").repeat(repeats, 1)))
        # self.error_stats.contributions = torch.cat((self.error_stats.contributions, torch.zeros((densification_mask.sum(), 1), device="cuda").repeat(repeats, 1)))
        # self.error_stats.areas = torch.cat((self.error_stats.areas, torch.ones((densification_mask.sum(), 1), device="cuda").repeat(repeats, 1)))
        self.error_stats.errors = torch.cat((self.error_stats.errors, self.error_stats.errors[densification_mask].repeat(repeats, 1)))
        self.error_stats.contributions = torch.cat((self.error_stats.contributions, self.error_stats.contributions[densification_mask].repeat(repeats, 1)))
        self.error_stats.areas = torch.cat((self.error_stats.areas, self.error_stats.areas[densification_mask].repeat(repeats, 1)))

        self.density_gradient_accum = torch.zeros((self.num_primitives, 1), device="cuda")
        torch.cuda.empty_cache()

    def prune(self,
              optimizer,
              min_opacity: float,
              extent: float,
              prune_mask: torch.Tensor | None = None):
        
        if isinstance(prune_mask, torch.Tensor):
            prune_mask = torch.logical_or(prune_mask.view(-1), (self.get_opacity < min_opacity).view(-1))
        else:
            prune_mask = (self.get_opacity < min_opacity).view(-1)
        self.prune_points(prune_mask, optimizer)

    def produce_clusters(self, num_clusters=256, store_dict_path=None):
        max_coeffs_num = (self.max_sh_degree + 1)**2 - 1
        codebook_dict = OrderedDict({})

        codebook_dict["features_dc"] = generate_codebook(self._features_dc.detach()[:, 0],
                                                         num_clusters=num_clusters, tol=0.001)
        for sh_degree in range(max_coeffs_num):
                codebook_dict[f"features_rest_{sh_degree}"] = generate_codebook(
                    self._features_rest.detach()[:, sh_degree], num_clusters=num_clusters)

        codebook_dict["opacity"] = generate_codebook(self.get_opacity.detach(),
                                                     self.inverse_opacity_activation, num_clusters=num_clusters)
        codebook_dict["scaling"] = generate_codebook(self.get_scaling.detach(),
                                                     self.scaling_inverse_activation, num_clusters=num_clusters)
        codebook_dict["rotation_re"] = generate_codebook(self.get_rotation.detach()[:, 0:1],
                                                         num_clusters=num_clusters)
        codebook_dict["rotation_im"] = generate_codebook(self.get_rotation.detach()[:, 1:],
                                                         num_clusters=num_clusters)
        if store_dict_path is not None:
            torch.save(codebook_dict, os.path.join(store_dict_path, 'codebook.pt'))
        
        self._codebook_dict = codebook_dict

    @torch.no_grad()
    # Decreases the texel_pixel ratio of the selected primitives,
    # resulting in a upsampling of the texture
    # TODO Have a optimizer-less version (GaussianModel self contained)
    def increase_texture_resolution(self, optimizer, selection_mask: torch.Tensor, factor=2):
        new_texture_resolution = self._texture_map._sizes.clone()
        new_texture_resolution[selection_mask] *= factor
        new_texture_map = JaggedTensor(new_texture_resolution)
        
        new_texture_map.central_crop(self._texture_map)
        added_memory: int = 0
        for curr_tex_res, curr_boolean_mask in self._texture_map._iter():
            curr_mask_boolean = torch.logical_and(curr_boolean_mask, selection_mask)
            curr_mask = curr_mask_boolean.nonzero().int().squeeze(-1)
            if curr_mask.shape[0] == 0:
                continue

            jagged_mask = self._texture_map.create_jagged_mask(curr_mask)
            curr_texture_map = self.texture_map_activation(self._texture_map._values)[jagged_mask].view(-1, int(curr_tex_res[0]), int(curr_tex_res[1]), 3)

            upsampled_tex_map = JaggedTensor(
                new_texture_map._sizes[curr_mask], 
                self.inverse_texture_map_activation(torch.nn.functional.interpolate(curr_texture_map.permute(0, 3, 1, 2), mode="nearest", scale_factor=factor).permute(0, 2, 3, 1).reshape(-1, 3)))

            source_mask = torch.arange(curr_mask.shape[0], device="cuda", dtype=torch.int32)
            new_texture_map.central_crop(upsampled_tex_map, source_mask, curr_mask)

            added_memory += int(torch.prod(new_texture_map._sizes[curr_mask] - self._texture_map._sizes[curr_mask], dim=1).sum().item())

            self._texel_pixel_ratio[curr_mask] -= 1

        new_texture_map._values.nan_to_num_(0, 0, 0)
        self._texture_map._values = optimizer.replace_tensor_to_optimizer(
            new_texture_map._values,
            "texture_map",
            (
                self._texture_map.create_jagged_mask(~selection_mask),
                new_texture_map.create_jagged_mask(~selection_mask)
            )
            )["texture_map"]
        self._texture_map._sizes = new_texture_map._sizes

    @torch.no_grad()
    # TODO Have a optimizer-less version (GaussianModel self contained)
    def decrease_texture_resolution(
        self,
        optimizer,
        activation_mask: torch.Tensor,
        downscale_threshold: float = 0.02):

        texture_resolution = self._texture_map._sizes
        downsampled_mask_boolean = torch.zeros_like(activation_mask)

        downsampled_texture_resolution = self._texture_map._sizes[activation_mask] // 2
        downsampled_texture_map = JaggedTensor(downsampled_texture_resolution)

        loss = torch.zeros(int(activation_mask.sum()), device="cuda")

        for curr_tex_res in texture_resolution.unique(dim=0):
            curr_original_mask_boolean = torch.logical_and(
                (texture_resolution == curr_tex_res).all(dim=1),
                activation_mask)
            curr_mask = curr_original_mask_boolean.nonzero().int().squeeze(-1)
            if curr_tex_res.min() == 1 or curr_mask.shape[0] == 0:
                continue

            curr_downsampled_mask_boolean = (downsampled_texture_map._sizes == curr_tex_res//2).all(dim=1)
            downsampled_jagged_mask = downsampled_texture_map.create_jagged_mask(curr_downsampled_mask_boolean)

            # Create the downscaled and reconstructed versions of the texture maps
            (original,
             downscaled,
             reconstructed) = self._texture_map.generate_downscaled_reconstructed_maps(curr_mask, self.texture_map_activation)
            
            # Store the downsampled version in case we use it
            downsampled_texture_map._values[downsampled_jagged_mask] = self.inverse_texture_map_activation(downscaled.reshape(-1, 3))

            difference = (reconstructed - original).mean(dim=-1)

            # We also weight the maps with the falloff value that they encounter
            clamped_kernels = gaussian_kernel_2d(curr_tex_res, self.get_scaling[curr_mask], self.texel_size[curr_mask], cutoff=self._texture_cutoff)

            weight_sum = clamped_kernels.sum(dim=[1, 2])
            weight_sum[weight_sum.isclose(torch.zeros_like(weight_sum))] = 1.


            loss[curr_downsampled_mask_boolean] = (difference * clamped_kernels).abs().sum(dim=[1, 2]) / weight_sum


        original_indices_mask = torch.arange(self._texture_map.n_tensors, device="cuda")
        original_mask_boolean = torch.zeros(self._texture_map.n_tensors, dtype=torch.bool, device="cuda")

        downsampled_mask_boolean = loss < downscale_threshold

        if downsampled_mask_boolean.sum() == 0:
            return

        downsampled_texture_map._values.nan_to_num_(0, 0, 0)

        original_indices_mask = original_indices_mask[activation_mask][downsampled_mask_boolean]
        original_mask_boolean[original_indices_mask] = True
        
        self._texel_pixel_ratio[original_indices_mask] += 1

        new_texture_resolution = self._texture_map._sizes.clone()
        new_texture_resolution[original_indices_mask] //= 2
        new_texture_map = JaggedTensor(new_texture_resolution)
        new_texture_map.central_crop(self._texture_map)
        new_texture_map.central_crop(downsampled_texture_map, downsampled_mask_boolean, original_indices_mask)

        self._texture_map._values = optimizer.replace_tensor_to_optimizer(
            new_texture_map._values,
            "texture_map",
            (
                self._texture_map.create_jagged_mask(~original_mask_boolean),
                new_texture_map.create_jagged_mask(~original_mask_boolean)
            )
            )["texture_map"]

        self._texture_map._sizes = new_texture_map._sizes
    
    @torch.no_grad()
    def split_textured_primitives(self,
                                  optimizer,
                                  splitting_threshold: torch.Tensor,
                                  direction_idx: int = 0,
                                  percentile: float = 0,
                                  max_points: int = -1) -> torch.Tensor:
        texture_resolution = self._texture_map._sizes
        # Get the actively used texture size to calculate clones translations
        active_texture_resolution = self._calculate_active_texture_resolution(return_differentiable=True).clamp_max(texture_resolution)
        texture_world_extent = self.texel_size * active_texture_resolution

        rotations = build_rotation(self.get_rotation)

        # Shape N x 3 x 2 directions
        tangent_vectors = rotations[:, :, :2]

        directions = torch.empty((0,2), device="cuda")
        
        # Find primitives that stretch beyond maximum texture resolution
        overflow_mask = active_texture_resolution[:, direction_idx] > splitting_threshold[direction_idx]
        
        # Find primitives with high error
        error_mask = (self.error_stats.errors > max(self.error_stats.errors.quantile(percentile).item(), 1e-6)).view(-1)

        # Split points that have high texture resolution and high error
        splitting_mask = torch.logical_and(overflow_mask, error_mask)

        if splitting_mask.sum() == 0 or (max_points != -1 and self.num_primitives >= max_points):
            return torch.zeros(self.num_primitives, device="cuda", dtype=torch.bool)

        # Make sure to not pass over the max_points threshold by limiting the 
        # primitives getting split
        if max_points != -1 and splitting_mask.sum() + self.num_primitives > max_points:
            indices = splitting_mask.nonzero()
            splitting_mask.zero_()
            splitting_mask[indices[:max_points - self.num_primitives]] = True
    
        # Create the offset for the new points
        # Get the directions +-[1,0] or +-[0,1] multiplied by a factor
        offset = 1
        magnitude = 1
        gaussian_falloff_at_offset = float(np.exp(-1/2*offset**2))
        
        directions = offset/6 * torch.tensor([[(1-direction_idx), direction_idx], [-(1-direction_idx), -direction_idx]], device="cuda")[:, None]
        corrected_directions = torch.zeros(int(splitting_mask.sum()) * 2, 1, 2, device="cuda")
        new_texture_resolution = texture_resolution[splitting_mask].repeat((2, 1))
        new_texture_map = JaggedTensor(new_texture_resolution)

        # Iterate over the tensor inside the jagged tensor
        for curr_tex_res, boolean_mask in self._texture_map._iter():
            curr_res_mask = torch.logical_and(splitting_mask, boolean_mask)
            if curr_res_mask.sum() == 0:
                continue

            # Get the mask to filter the primitives of the new_textuer_map
            new_res_mask = (new_texture_map._sizes == torch.tensor([curr_tex_res[0], curr_tex_res[1]], device="cuda", dtype=torch.int32)).all(dim=1)
            
            new_center_shift = (-directions * active_texture_resolution[curr_res_mask]).int()
            
            # Crop the original texture maps at the dispalced locations
            new_texture_map.central_crop(
                self._texture_map,
                torch.arange(self._texture_map.n_tensors, device="cuda", dtype=torch.int32)[curr_res_mask].repeat(2),
                new_res_mask,
                new_center_shift.reshape(-1, 2).int())

            # Recalculate directions because of the quantisation in texture recentering
            corrected_directions[new_res_mask] = (-new_center_shift / active_texture_resolution[curr_res_mask]).view(-1, 1, 2)

        directions_world = tangent_vectors[splitting_mask].unsqueeze(1) * corrected_directions.view(2, -1, 2).swapdims(0,1).reshape(-1, 2, 1, 2)
        
        # This ugly line takes the original xyz and adds some offsets along the tangent vectors, multiplied by some magnitude and the texture's extent
        new_xyz = (self._xyz[splitting_mask].unsqueeze(-2).expand(-1, 2, -1) + (texture_world_extent[splitting_mask].view(-1, 1, 1, 2) * directions_world).sum(dim=-1)).permute(1,0,2).reshape(-1,3)
        new_features_dc = self._features_dc[splitting_mask].repeat((2, 1, 1))
        new_features_rest = self._features_rest[splitting_mask].repeat((2, 1, 1))
        
        new_opacities = self.inverse_opacity_activation(self.get_opacity[splitting_mask].repeat((2, 1)) * magnitude * torch.tensor([[gaussian_falloff_at_offset, gaussian_falloff_at_offset]], device="cuda").repeat(int(splitting_mask.sum()), 1).T.reshape(-1 ,1))
        scaling = self.get_scaling[splitting_mask].repeat((2, 1))
        scaling[:, direction_idx] = (scaling[:, direction_idx] / 2).clamp_max(splitting_threshold[direction_idx] * self.texel_size[splitting_mask].view(-1).repeat((2)) / 3)
        new_scaling = self.scaling_inverse_activation(scaling)
        new_rotation = self._rotation[splitting_mask].repeat((2, 1))

        self.densification_postfix(optimizer,
                                   new_xyz,
                                   new_features_dc, 
                                   new_features_rest, 
                                   new_texture_map._values, 
                                   new_texture_resolution, 
                                   new_opacities, 
                                   new_scaling, 
                                   new_rotation, 
                                   densification_mask = splitting_mask,
                                   repeats = 2)
        
        return torch.cat((splitting_mask, torch.zeros(new_xyz.shape[0], device="cuda", dtype=torch.bool)))

    @torch.no_grad()
    def initialise_texel_pixel_ratio(self, initial_texture_resolution: torch.Tensor):
        """Initializes texel_pixel_ratio

        This function sets the texel_pixel_ratio by taking as input either a preloaded model, in which case _texel_size will be prefilled
        or a new model that it is wanted to match the given resolution.
        """
        
        # if we have a loaded model
        if self._texel_size is not None:
            self._texel_pixel_ratio = torch.log2(self._texel_size / self._pixel_size)
            self._texel_size = None
            return

        assert self._pixel_size.max() < 10000., "Unseen primitives still have the initial pixel size value for some reason"
        new_texel_size = ((self._calculate_active_texture_resolution(return_differentiable=True).detach() / initial_texture_resolution) * self.texel_size).min(dim=1, keepdim=True).values
        new_texel_pixel_ratio = torch.log2(new_texel_size / self._pixel_size).ceil().int()
        self._texel_pixel_ratio = new_texel_pixel_ratio.view(-1, 1)
        self._texel_pixel_ratio.clamp_min_(1)

    @torch.no_grad()
    # TODO Have a optimizer-less version (GaussianModel self contained)
    def texture_map_resize(self, optimizer, powers_of_two: bool = True):
        """Checks if texture map needs resizing by computing the new resolutions and calling the resizing CUDA kernel"""

        new_texture_map = JaggedTensor(self._calculate_active_texture_resolution(powers_of_two=powers_of_two).int())
        update_mask = (new_texture_map._sizes != self._texture_map._sizes).any(dim=1)

        if update_mask.sum() == 0:
            return
        new_texture_map.central_crop(self._texture_map)

        new_texture_map._values.nan_to_num_(0, 0, 0)

        self._texture_map._values = optimizer.replace_tensor_to_optimizer(
            new_texture_map._values,
            "texture_map",
            (
                self._texture_map.create_jagged_mask(~update_mask),
                new_texture_map.create_jagged_mask(~update_mask)
            )
            )["texture_map"]
        self._texture_map._sizes = new_texture_map._sizes
        torch.cuda.empty_cache()
