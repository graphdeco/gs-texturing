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

#include <cuda.h>
#include <cuda_runtime.h>
// #define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace ErrorStats {
    struct InputImages {
        const float *gt_image;
        const float *render;
    };

    struct ViewErrorStats {
        int32_t   *n_pixels;
        float     *contributions;
        float     *errors;
        glm::vec2 *first_moments;
        glm::vec3 *second_moments;
    };
}
