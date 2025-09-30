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

#include "error_stats.h"

#include <cooperative_groups.h>

#include "auxiliary.h"
#include "utils/math.h"

namespace cg = cooperative_groups;

struct ThreadPixelInfo {
    uint2     pix_min;
    uint2     pix_max;
    uint2     pix;
    uint32_t  pix_id;
    glm::vec2 pixf;
};

__device__ ThreadPixelInfo thread_tile_info(cg::thread_block block, int width, int height) {
    dim3 dim_threads  = block.dim_threads();
    dim3 group_index  = block.group_index();
    dim3 thread_index = block.thread_index();

    ThreadPixelInfo ret;
    ret.pix_min = {
        group_index.x * dim_threads.x,
        group_index.y * dim_threads.y
    };
    ret.pix_max = {
        min(ret.pix_min.x + dim_threads.x, width),
        min(ret.pix_min.y + dim_threads.y, height)
    };
    ret.pix = {
        ret.pix_min.x + thread_index.x,
        ret.pix_min.y + thread_index.y
    };
    ret.pix_id = width * ret.pix.y + ret.pix.x;
    ret.pixf   = {(float)ret.pix.x, (float)ret.pix.y};

    return ret;
}

__device__ void copy_pixel(
    const float *__restrict__ image,
    int      width,
    int      height,
    int      nChannels,
    uint32_t pix_id,
    float *__restrict__ out_pixel
) {
    uint32_t stride = width * height;

    for (int c = 0; c < nChannels; c++)
        out_pixel[c] = image[c * stride + pix_id];
}

__device__ void compute_pixel_errors(
    const float *__restrict__ gt_pixel,
    const float *__restrict__ render_pixel,
    float *__restrict__ error_per_channel,
    float &__restrict__ error,
    int nChannels
) {
    error = 0;
    for (int c = 0; c < nChannels; c++) {
        float clamped_render_pixel = min(max(render_pixel[c], 0.f), 1.f);
        error_per_channel[c] = abs(gt_pixel[c] - clamped_render_pixel);
        error += error_per_channel[c];
    }
    error /= nChannels;
}

__device__ void load_batch_in_shared_memory(
    int   batch,
    int   batch_size,
    int   thread_rank,
    uint2 range,
    const uint32_t *__restrict__ point_list,
    const glm::vec3 *__restrict__ p_view,
    const glm::vec2 *__restrict__ scale,
    const float *__restrict__ opacity,
    const glm::vec4 *__restrict__ rotation,
    const glm::vec3 *__restrict__ normal,
    const float *__restrict__ weight_precomputed,
    uint32_t *__restrict__ collected_id,
    glm::vec3 *__restrict__ collected_p_view,
    glm::vec2 *__restrict__ collected_scale,
    float *__restrict__ collected_opacity,
    glm::vec4 *__restrict__ collected_rotation,
    glm::vec3 *__restrict__ collected_normal,
    float *__restrict__ collected_weight_precomputed,
    int nChannels
) {
    auto progress = batch * batch_size + thread_rank;
    if (range.x + progress >= range.y) return;

    const int coll_id                    = point_list[range.x + progress];
    collected_id[thread_rank]            = coll_id;
    collected_p_view[thread_rank]        = p_view[coll_id];
    collected_scale[thread_rank]         = scale[coll_id];
    collected_opacity[thread_rank]       = opacity[coll_id];
    collected_rotation[thread_rank]      = rotation[coll_id];
    collected_normal[thread_rank]        = normal[coll_id];
    collected_weight_precomputed[thread_rank] = weight_precomputed[coll_id];
}

__device__ void accumulate_errors_and_moments(
    const glm::vec2 &__restrict__ pixel_center,
    const int width, const int height,
    const float2 focal,
    const glm::vec3 &curr_p_view,
    const glm::vec2 &curr_scale,
    const float curr_opacity,
    const glm::vec3 &curr_normal,
    const glm::vec4 &curr_rotation,
    const float* viewmatrix,
    const float &__restrict__ weight_precomputed,
    float &__restrict__ T,
    bool &__restrict__ done,
    const float pixel_error,
    int32_t &__restrict__ n_pixels,
    float &__restrict__ contribution,
    float &__restrict__ gaussian_error
) {
    // Ray view-space
	const glm::vec3 ray_origin_view_space(0.f);
	// Taking a point with t2=1 (third coordinate in view space) we get
	const glm::vec3 ray_direction_view = glm::normalize(glm::vec3((pixel_center.x - (width - 1) / 2.f) / focal.x, (pixel_center.y - (height - 1) / 2.f) / focal.y, 1.0f));

    const glm::mat3 R = buildRotationMatrix(curr_rotation);

    // R is orthonormal, so R^-1 = R^T
    const glm::mat3 inv_R = glm::transpose(R);

    const glm::mat3 &world2view = glm::mat3(
        viewmatrix[0], viewmatrix[1], viewmatrix[2],
        viewmatrix[4], viewmatrix[5], viewmatrix[6],
        viewmatrix[8], viewmatrix[9], viewmatrix[10]);
    
    // world2View is orthonormal, so world2View^-1 = world2View^T
    const glm::mat3 &inv_world2View = glm::transpose(world2view);

    // x = ray_origin + ray
    // <(x - p_view), normal> = 0
    const float ray_depth = calculateIntersectionRayDepth(
        ray_origin_view_space,
        ray_direction_view,
        curr_p_view,
        curr_normal);

    // TODO ADD NEAR_PLANE HERE
    if (ray_depth < 0.2f)
        return;

    const glm::vec3 intersection_point_view = ray_origin_view_space + ray_depth * ray_direction_view - curr_p_view;

    glm::vec2 intersection_point_canonical(0.f);
    glm::vec2 intersection_point_axisaligned(0.f);
    intersection_point_axisaligned = glm::vec2(inv_R * inv_world2View * intersection_point_view);
    intersection_point_canonical = 1.f / curr_scale * intersection_point_axisaligned;
    float base = glm::dot(intersection_point_canonical, intersection_point_canonical);
    
    float power = -0.5 * base;
    if (power > 0.0f)
        return;

    float alpha = min(0.99f, curr_opacity * exp(power));
    if (alpha < 1.0f / 255.0f)
        return;

    float test_T = T * (1 - alpha);
    if (test_T < 0.0001f) {
        done = true;
        return;
    }

    atomicAdd(&n_pixels, 1);

    // float weight = weight_precomputed < 0 ? alpha * T : weight_precomputed;
    float weight = alpha * T;
    atomicAdd(&contribution, weight);

    float error_weight = pixel_error * weight;
    atomicAdd(&gaussian_error, error_weight);

    T = test_T;
}

template <uint32_t nChannels>
__global__ void __launch_bounds__(BLOCK_X *BLOCK_Y) renderCUDA(
    const uint2 *__restrict__ ranges,
    const uint32_t *__restrict__ point_list,
    int width,
    int height,
    const float2 focal,
    const float* viewmatrix,
    const float *__restrict__ bg_color,
    const glm::vec3 *__restrict__ p_view,
    const glm::vec2 *__restrict__ scale,
    const float *__restrict__ opacity,
    const glm::vec4 *__restrict__ rotation,
    const glm::vec3 *__restrict__ normal,
    const float *__restrict__ weight_precomputed,
    const float *__restrict__ final_Ts,
    const uint32_t *__restrict__ n_contrib,
    const float *__restrict__ loss_img,
    int32_t *__restrict__ n_pixels,
    float *__restrict__ contributions,
    float *__restrict__ errors
) {
    // Block info
    auto grid        = cg::this_grid();
    auto block       = cg::this_thread_block();
    auto block_size  = block.size();
    auto thread_rank = block.thread_rank();

    // Info about the pixel handled by the thread
    const ThreadPixelInfo pixelInfo = thread_tile_info(block, width, height);
    
    uint2     pix_min = pixelInfo.pix_min;
    uint2     pix_max = pixelInfo.pix_max;
    uint2     pix = pixelInfo.pix;
    uint32_t  pix_id = pixelInfo.pix_id;
    glm::vec2 pixf = pixelInfo.pixf;

    const bool inside = pix.x < width && pix.y < height;

    // The gaussians that are splatted on the tile
    const uint2 range = ranges[grid.block_rank()];

    float loss_pixel[nChannels];
    float error(0.f);
    if (inside) {
        copy_pixel(loss_img, width, height, nChannels, pix_id, loss_pixel);
        error = loss_pixel[0];
    }

    // Initialize helper variables
    bool  done = !inside;
    float T    = 1.0f;

    // Gaussians will be loaded in batches in shared memory
    constexpr uint32_t  batch_size = BLOCK_SIZE;
	// Allocate storage for batches of collectively fetched data.
	__shared__ uint32_t collected_id[batch_size];
	__shared__ glm::vec3 collected_p_view[batch_size];
	__shared__ glm::vec2 collected_scale[batch_size];
	__shared__ float collected_opacity[batch_size];
	__shared__ glm::vec3 collected_normal[batch_size];
    __shared__ glm::vec4 collected_rotation[batch_size];
    __shared__ float collected_weight_precomputed[batch_size];
    
    int       n_remaining = range.y - range.x;
    const int n_batches   = ((range.y - range.x + batch_size - 1) / batch_size);

    // Traverse all Gaussians
    for (int batch = 0; batch < n_batches; batch++, n_remaining -= batch_size) {
        block.sync();
        load_batch_in_shared_memory(
            batch,
            batch_size,
            thread_rank,
            range,
            point_list,
            p_view,
            scale,
            opacity,
            rotation,
            normal,
            weight_precomputed,
            collected_id,
            collected_p_view,
            collected_scale,
            collected_opacity,
            collected_rotation,
            collected_normal,
            collected_weight_precomputed,
            nChannels
        );
        block.sync();

        // Iterate over Gaussians in batch
        for (int i = 0; !done && i < min(batch_size, n_remaining); i++) {
            auto gaussian_id = collected_id[i];

            accumulate_errors_and_moments(
                pixf,
                width, height,
                focal,
                collected_p_view[i],
                collected_scale[i],
                collected_opacity[i],
                collected_normal[i],
                collected_rotation[i],
                viewmatrix,
                collected_weight_precomputed[i],
                T,
                done,
                error,
                n_pixels[gaussian_id],
                contributions[gaussian_id],
                errors[gaussian_id]);
        }
    }
}

void ErrorStats::render(
    dim3                                     grid,
    dim3                                     block,
    const uint2                             *ranges,
    const uint32_t                          *point_list,
    int                                      width,
    int                                      height,
    const float2                             focal,
    const float                             *viewmatrix,
    const float                             *bg_color,
    const CudaRasterizer::GeometryState     &geometry_state,
    const CudaRasterizer::TextureState      &texture_state,
    const float                             *weight_precomputed,
    const float                             *final_Ts,
    const uint32_t                          *n_contrib,
    const float                             *loss_img,
    ViewErrorStats                           view_error_stats
) {
    renderCUDA<1><<<grid, block>>>(
        ranges,
        point_list,
        width,
        height,
        focal,
        viewmatrix,
        bg_color,
        (glm::vec3 *)texture_state.buffer.mean,
        geometry_state.scale,
        geometry_state.opacity,
        geometry_state.rotation,
        texture_state.buffer.normal,
        weight_precomputed,
        final_Ts,
        n_contrib,
        loss_img,
        view_error_stats.n_pixels,
        view_error_stats.contributions,
        view_error_stats.errors
    );
}

__global__ void __launch_bounds__(BLOCK_X *BLOCK_Y) remove_pixel_scalingCUDA(
    float focal_x,
    float focal_y,
    glm::vec2 *__restrict__ view_first_moments,
    glm::vec3 *__restrict__ view_second_moments,
    int n_gaussians
) {
    // Thread info
    auto grid        = cg::this_grid();
    auto thread_rank = grid.thread_rank();

    if (thread_rank >= n_gaussians) return;

    // Transform diagonal of the inverse of the affine approximation's jacobian
    glm::vec2 vector_pixel_scaling = {focal_x, focal_y};

    auto &view_first_moment = view_first_moments[thread_rank];
    view_first_moment /= vector_pixel_scaling;

    auto &view_second_moment = view_second_moments[thread_rank];
    view_second_moment /= packed_self_outer_product(vector_pixel_scaling);
}

void ErrorStats::remove_pixel_scaling(
    float      focal_x,
    float      focal_y,
    glm::vec2 *view_first_moments,
    glm::vec3 *view_second_moments,
    int        n_gaussians
) {
    remove_pixel_scalingCUDA<<<(n_gaussians + 255) / 256, 256>>>(
        focal_x,
        focal_y,
        view_first_moments,
        view_second_moments,
        n_gaussians
    );
}
