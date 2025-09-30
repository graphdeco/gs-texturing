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

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(*args):
    return _RasterizeGaussians.apply(*args)

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        degree,
        colors_precomp,
        opacities,
        scales,
        rotations,
        texture_map,
        texture_resolution,
        texture_map_start_offset,
        texel_size,
        cov3Ds_precomp,
        raster_settings,
        flags
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg, 
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            texture_map,
            texture_resolution,
            texture_map_start_offset,
            texel_size,
            sh,
            degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            flags["colour_type"],
            flags["calculate_weight_precomputed"],
            raster_settings.debug,
            flags["texture_debug_view"]
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                num_rendered, color, out_features, out_weight, radii, textureBuffer, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, out_features, out_weight, radii, textureBuffer, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.degree = degree
        ctx.save_for_backward(colors_precomp,
                              means3D, 
                              scales, 
                              opacities, 
                              rotations, 
                              cov3Ds_precomp, 
                              radii, 
                              sh, 
                              texture_map, 
                              texture_resolution, 
                              texture_map_start_offset, 
                              texel_size, 
                              textureBuffer,
                              geomBuffer,
                              binningBuffer,
                              imgBuffer)
        ctx.flags = flags
        return color, radii, out_features, out_weight

    @staticmethod
    def backward(ctx, grad_out_color, _, grad_out_features, unused_out_weight_grad):
        grad_out_features = grad_out_features[:4]
        # Restore necessary values from context
        degree = ctx.degree
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        flags = ctx.flags
        (colors_precomp,
        means3D,
        scales,
        opacities,
        rotations,
        cov3Ds_precomp,
        radii,
        sh,
        texture_map,
        texture_resolution,
        texture_map_start_offset,
        texel_size,
        textureBuffer,
        geomBuffer,
        binningBuffer,
        imgBuffer) = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D, 
                radii, 
                colors_precomp, 
                scales,
                opacities,
                rotations,
                cov3Ds_precomp, 
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                texture_map,
                texture_resolution,
                texture_map_start_offset,
                texel_size,
                grad_out_color,
                grad_out_features,
                sh,
                degree, 
                raster_settings.campos,
                textureBuffer,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_colors_precomp, grad_opacities, grad_means3D, grad_sh, grad_scales, grad_rotations, grads_texture_map = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
            grad_colors_precomp, grad_opacities, grad_means3D, grad_sh, grad_scales, grad_rotations, grads_texture_map = _C.rasterize_gaussians_backward(*args)
        
        grad_means2D = torch.zeros((grad_means3D.shape[0], 2), device=grad_means3D.device)
        grad_cov3Ds_precomp = torch.zeros((grad_means3D.shape[0], 6), device=grad_means3D.device)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            None,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grads_texture_map,
            None,
            None,
            None,
            grad_cov3Ds_precomp,
            None,
            None,
        )

        return grads

    @staticmethod
    def forward_with_full_state(
        means3D,
        sh,
        degree,
        colors_precomp,
        opacities,
        scales,
        rotations,
        texture_map,
        texture_resolution,
        texture_map_start_offset,
        texel_size,
        cov3Ds_precomp,
        raster_settings,
        flags
    ):
        # Restructure arguments the way that the C++ lib expects them
        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg, 
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            texture_map,
            texture_resolution,
            texture_map_start_offset,
            texel_size,
            sh,
            degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            flags["colour_type"],
            flags["calculate_weight_precomputed"],
            raster_settings.debug,
            flags["texture_debug_view"]
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            # Copy them before they can be corrupted
            cpu_args = cpu_deep_copy_tuple(args)

            try:
                return _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print(
                    "\nAn error occured in forward. Please forward snapshot_fw.dump for debugging."
                )
                raise ex
        else:
            return _C.rasterize_gaussians(*args)


class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self,
                means3D,
                means2D,
                opacities,
                shs = None,
                degree = None,
                colors_precomp = None,
                scales = None,
                rotations = None,
                texture_map = None,
                texture_resolution = None,
                texture_map_start_offset = None,
                texel_size = None,
                cov3D_precomp = None,
                flags = None):
        
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            degree,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            texture_map,
            texture_resolution,
            texture_map_start_offset,
            texel_size,
            cov3D_precomp,
            raster_settings, 
            flags
        )

    def forward_with_full_state(self, 
                                means3D,
                                shs = None,
                                degree = None,
                                colors_precomp = None,
                                opacities = None,
                                scales = None,
                                rotations = None,
                                texture_map = None,
                                texel_size = None,
                                texture_resolution = None,
                                texture_map_start_offset = None,
                                cov3D_precomp = None,
                                raster_settings = None,
                                flags = None):
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')

        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')

        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        return _RasterizeGaussians.forward_with_full_state(
            means3D,
            shs,
            degree,
            colors_precomp,
            opacities,
            scales,
            rotations,
            texture_map,
            texture_resolution,
            texture_map_start_offset,
            texel_size,
            cov3D_precomp,
            raster_settings,
            flags
        )

    def error_stats(
        self,
        means3D,
        opacity,
        shs,
        colors_precomp,
        scales,
        rotations,
        radii,
        cov3D_precomp,
        geom_buffer,
        texture_buffer,
        num_rendered,
        viewmatrix,
        binning_buffer,
        img_buffer,
        weight_precomputed,
        loss_img,
    ):
        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')

        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')

        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        raster_settings = self.raster_settings

        args = (
            raster_settings.bg,
            means3D,
            opacity,
            scales,
            rotations,
            colors_precomp,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            geom_buffer,
            texture_buffer,
            num_rendered,
            viewmatrix,
            binning_buffer,
            img_buffer,
            weight_precomputed,
            loss_img,
            raster_settings.debug,
        )

        n_pixels, contributions, errors, first_moments, second_moments = (
            _C.rasterize_gaussians_error_stats(*args)
        )

        return {
            "n_pixels": n_pixels,
            "contributions": contributions,
            "errors": errors,
            "first_moments": first_moments,
            "second_moments": second_moments
        }