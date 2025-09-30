/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#pragma once

#include <cooperative_groups.h>

#include "auxiliary.h"
#include "error_stats_types.h"
#include "rasterizer_impl.h"

namespace ErrorStats {

    void render(
        dim3             grid,
        dim3             block,
        const uint2     *ranges,
        const uint32_t  *point_list,
        int              width,
        int              height,
        const float2     focal,
        const float     *viewmatrix,
        const float     *bg_color,
        const CudaRasterizer::GeometryState   &geometry_state,
        const CudaRasterizer::TextureState    &texture_state,
        const float     *weight_precomputed,
        const float     *final_Ts,
        const uint32_t  *n_contrib,
        const float     *loss_img,
        ViewErrorStats   view_error_stats
    );

    void remove_pixel_scaling(
        float      focal_x,
        float      focal_y,
        glm::vec2 *view_first_moments,
        glm::vec3 *view_second_moments,
        int        n_gaussians
    );
}
