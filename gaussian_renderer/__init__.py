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
import math
from diff_gaussian_rasterization_texture import GaussianRasterizationSettings, GaussianRasterizer
from utils.sh_utils import eval_sh

def _compute_rasterization_attirbutes(camera, pc, scaling_modifier, pipe, override_color):
    means3D = pc.get_xyz
    opacities = pc.get_opacity
    texture_map = pc.get_texture_map
    texture_resolution = pc._texture_map._sizes
    texture_map_start_offset = pc._texture_map.start_offsets
    texel_size = pc.texel_size

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = (pc.get_scaling * scaling_modifier)[:, :2].contiguous()
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Set up rasterization configuration
    tanfovx = math.tan(camera.FoVx * 0.5)
    tanfovy = math.tan(camera.FoVy * 0.5)

    return (means3D,
            opacities,
            scales,
            rotations,
            cov3D_precomp,
            shs,
            texture_map,
            texture_resolution,
            texture_map_start_offset,
            texel_size,
            pc.max_sh_degree,
            pc.active_sh_degree,
            colors_precomp,
            tanfovx,
            tanfovy)

def render(viewpoint_camera,
           pc,
           pipe,
           bg_color : torch.Tensor,
           scaling_modifier = 1.0,
           override_color = None,
           measure_fps=False,
           texture_debug_view=False,
           colour_type="full",
           mask=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    if colour_type == "full":
        colour_type = 0
    elif colour_type == "base":
        colour_type = 1
    elif colour_type == "highlights":
        colour_type = 2

    (
        means3D,
        opacity,
        scales,
        rotations,
        cov3D_precomp,
        shs,
        texture_map,
        texture_resolution,
        texture_map_start_offset,
        texel_size,
        max_sh_degree,
        active_sh_degree,
        colors_precomp,
        tanfovx,
        tanfovy,
    ) = _compute_rasterization_attirbutes(
        viewpoint_camera, pc, scaling_modifier, pipe, override_color
    )


    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda")[:, :2] + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        # debug=True
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means2D = screenspace_points

    # per_band_count = torch.tensor(pc.per_band_count, device="cuda", dtype=torch.int)
    # cumsum_count = torch.cumsum(per_band_count,dim=0).to(dtype=torch.int)
    # coeffs_num = torch.tensor([i*i for i in range(1, len(pc.per_band_count) + 1)], device="cuda", dtype=torch.int)
    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    if mask is not None:
        means3D = means3D[mask]
        means2D = means2D[mask]
        shs = shs[mask]
        opacity = opacity[mask]
        scales = scales[mask]
        # texture_map = texture_map[mask]
        texture_resolution = texture_resolution[mask]
        texture_map_start_offset = texture_map_start_offset[mask]
        texel_size = texel_size[mask]
        rotations = rotations[mask]
    fps = 0
    if measure_fps:
        start_timer = torch.cuda.Event(enable_timing=True)
        end_timer = torch.cuda.Event(enable_timing=True)
        start_timer.record()
    rendered_image, radii, features_image, out_weight = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        degree = active_sh_degree,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        texture_map = texture_map,
        texture_resolution = texture_resolution,
        texture_map_start_offset = texture_map_start_offset,
        texel_size = texel_size,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
        flags={
            "calculate_weight_precomputed": True,
            "texture_debug_view": texture_debug_view,
            "colour_type": colour_type,
        })
    normal_image = (features_image[0:3].permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)
    invdepth_image = features_image[3:4]
    if measure_fps:
        end_timer.record()
        torch.cuda.synchronize()
        start_timer.elapsed_time(end_timer)
        fps = 1 / (start_timer.elapsed_time(end_timer))

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "normal": normal_image,
            "out_weight": out_weight,
            "invdepth": invdepth_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "FPS": fps}

def render_with_full_state(
    viewpoint_camera,
    pc,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    (
        means3D,
        opacity,
        scales,
        rotations,
        cov3D_precomp,
        shs,
        texture_map,
        texture_resolution,
        texture_map_start_offset,
        texel_size,
        max_sh_degree,
        active_sh_degree,
        colors_precomp,
        tanfovx,
        tanfovy,
    ) = _compute_rasterization_attirbutes(
        viewpoint_camera, pc, scaling_modifier, pipe, override_color
    )

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    num_rendered, color, out_features, out_weights, radii, texture_buffer, geom_buffer, binning_buffer, img_buffer = rasterizer.forward_with_full_state(
            means3D = means3D,
            shs = shs,
            degree = active_sh_degree,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            texture_map = texture_map,
            texture_resolution = texture_resolution,
            texture_map_start_offset = texture_map_start_offset,
            texel_size = texel_size,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp,
            flags={
                "calculate_weight_precomputed": True,
                "texture_debug_view": False,
                "colour_type": False
            }
    )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "render": color,
        "means3D": means3D,
        "opacity": opacity,
        "scales": scales,
        "rotations": rotations,
        "cov3Ds_precomp": cov3D_precomp,
        "shs": shs,
        "colors_precomp": colors_precomp,
        "radii": radii,
        "raster_settings": raster_settings,
        "geom_buffer": geom_buffer,
        "num_rendered": num_rendered,
        "texture_buffer": texture_buffer,
        "binning_buffer": binning_buffer,
        "img_buffer": img_buffer,
        "weight_precomputed": out_weights
    }
