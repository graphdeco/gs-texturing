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

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include "cuda_rasterizer/utils.h"
#include <fstream>
#include <string>
#include <functional>
//#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "cuda_rasterizer/error_stats_types.h"


#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#include "cuda_rasterizer/auxiliary.h"
using namespace torch::indexing;

std::function<char *(size_t N)> resizeFunctional(torch::Tensor &t)
{
	auto lambda = [&t](size_t N)
	{
		t.resize_({(long long)N});
		return reinterpret_cast<char *>(t.contiguous().data_ptr());
	};
	return lambda;
}

template <
    typename element_t,
    typename cast_to = element_t,
    typename         = enable_if_element_vector_types_compatible_t<element_t, cast_to>>
const cast_to *tensor_data_ptr(const torch::Tensor &t) {
    return reinterpret_cast<const cast_to *>(t.contiguous().data_ptr<element_t>());
}

template <
    typename element_t,
    typename cast_to = element_t,
    typename         = enable_if_element_vector_types_compatible_t<element_t, cast_to>>
cast_to *tensor_data_ptr(torch::Tensor &t) {
    return reinterpret_cast<cast_to *>(t.contiguous().data_ptr<element_t>());
}


std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor &background,
	const torch::Tensor &means3D,
	const torch::Tensor &colors,
	const torch::Tensor &opacity,
	const torch::Tensor &scales,
	const torch::Tensor &rotations,
	const torch::Tensor &cov3D_precomp,
	const torch::Tensor &viewmatrix,
	const torch::Tensor &projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const int image_height,
	const int image_width,
	const torch::Tensor &textureMap,
	const torch::Tensor &textureResolution,
	const torch::Tensor &textureMapStartingOffset,
	const torch::Tensor &texelSize,
	const torch::Tensor &sh,
	const int degree,
	const torch::Tensor &campos,
	const bool prefiltered,
	const int colourType,
	const bool calculate_mean_weight,
	const bool debug,
	const bool texture_debug_view)
{
	if (means3D.ndimension() != 2 || means3D.size(1) != 3)
	{
		AT_ERROR("means3D must have dimensions (num_points, 3)");
	}

	const int P = means3D.size(0);
	const int H = image_height;
	const int W = image_width;

	auto int_opts = means3D.options().dtype(torch::kInt32);
	auto float_opts = means3D.options().dtype(torch::kFloat32);

	torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
	torch::Tensor out_features = torch::full({3 + 1, H, W}, 0.0, float_opts);
	torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));

	torch::Device device(torch::kCUDA);
	torch::TensorOptions options(torch::kByte);
	torch::Tensor textureBuffer = torch::empty({0}, options.device(device));
	torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
	torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
	torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
	std::function<char *(size_t)> textureFunc = resizeFunctional(textureBuffer);
	std::function<char *(size_t)> geomFunc = resizeFunctional(geomBuffer);
	std::function<char *(size_t)> binningFunc = resizeFunctional(binningBuffer);
	std::function<char *(size_t)> imgFunc = resizeFunctional(imgBuffer);
	
	torch::Tensor out_touched_pixels = torch::full({P, 1}, 0, int_opts);
	torch::Tensor out_transmittance = torch::full({P, 1}, 1e-10f, float_opts);

	int rendered = 0;
	if (P != 0)
	{
		int M = 0;
		if (sh.size(0) != 0)
		{
			M = sh.size(1);
		}

		rendered = CudaRasterizer::Rasterizer::forward(
			textureFunc,
			geomFunc,
			binningFunc,
			imgFunc,
			P, degree, M,
			background.contiguous().data<float>(),
			W, H,
			means3D.contiguous().data<float>(),
			sh.contiguous().data_ptr<float>(),
			(float3 *)textureMap.contiguous().data<float>(),
			(int2 *)textureResolution.contiguous().data<int>(),
			textureMapStartingOffset.contiguous().data<int>(),
			texelSize.contiguous().data_ptr<float>(),
			colors.contiguous().data<float>(),
			opacity.contiguous().data<float>(),
			scales.contiguous().data_ptr<float>(),
			rotations.contiguous().data_ptr<float>(),
			cov3D_precomp.contiguous().data<float>(),
			viewmatrix.contiguous().data<float>(),
			projmatrix.contiguous().data<float>(),
			campos.contiguous().data<float>(),
			make_float2(tan_fovx, tan_fovy),
			prefiltered,
			out_color.contiguous().data<float>(),
			out_features.contiguous().data<float>(),
			out_touched_pixels.contiguous().data<int>(),
			out_transmittance.contiguous().data<float>(),
			static_cast<CudaRasterizer::ColourType>(colourType),
			radii.contiguous().data<int>(),
			calculate_mean_weight,
			debug,
			texture_debug_view);
	}
	if(calculate_mean_weight)
		out_transmittance /= out_touched_pixels.max(torch::ones({P, 1}, means3D.options()));

	return std::make_tuple(rendered, out_color, out_features, out_transmittance, radii, textureBuffer, geomBuffer, binningBuffer, imgBuffer);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansBackwardCUDA(
	const torch::Tensor &background,
	const torch::Tensor &means3D,
	const torch::Tensor &radii,
	const torch::Tensor &colors,
	const torch::Tensor &scales,
	const torch::Tensor &opacities,
	const torch::Tensor &rotations,
	const torch::Tensor &cov3D_precomp,
	const torch::Tensor &viewmatrix,
	const torch::Tensor &projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const torch::Tensor &textureMap,
	const torch::Tensor &textureResolution,
	const torch::Tensor &textureMapStartingOffset,
	const torch::Tensor &texelSize,
	const torch::Tensor &dL_dout_color,
	const torch::Tensor &dL_dout_features,
	const torch::Tensor &sh,
	const int degree,
	const torch::Tensor &campos,
	const torch::Tensor &textureBuffer,
	const torch::Tensor &geomBuffer,
	const int R,
	const torch::Tensor &binningBuffer,
	const torch::Tensor &imageBuffer,
	const bool debug)
{
	const int P = means3D.size(0);
	const int H = dL_dout_color.size(1);
	const int W = dL_dout_color.size(2);

	int M = 0;
	if (sh.size(0) != 0)
	{
		M = sh.size(1);
	}

	torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
	torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());
	// dL_dconic refers to the derivative in respect to the 3 components of the 
	// symmetric cov2D matrix
	torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
	torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
	torch::Tensor dL_dscales = torch::zeros({P, 2}, means3D.options());
	torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
	torch::Tensor dL_dtextureFeatures = torch::zeros({textureMap.size(0), 3}, means3D.options());

	if (P != 0)
	{
		CudaRasterizer::Rasterizer::backward(P, degree, M, R,
											 background.contiguous().data<float>(),
											 W, H,
											 means3D.contiguous().data<float>(),
											 sh.contiguous().data<float>(),
											 (float3 *)textureMap.contiguous().data<float>(),
											 (int2 *)textureResolution.contiguous().data<int>(),
											 textureMapStartingOffset.contiguous().data<int>(),
											 texelSize.contiguous().data_ptr<float>(),
											 colors.contiguous().data<float>(),
											 scales.contiguous().data_ptr<float>(),
											 opacities.contiguous().data_ptr<float>(),
											 rotations.contiguous().data_ptr<float>(),
											 cov3D_precomp.contiguous().data<float>(),
											 viewmatrix.contiguous().data<float>(),
											 projmatrix.contiguous().data<float>(),
											 campos.contiguous().data<float>(),
											 make_float2(tan_fovx, tan_fovy),
											 radii.contiguous().data<int>(),
											 reinterpret_cast<char *>(textureBuffer.contiguous().data_ptr()),
											 reinterpret_cast<char *>(geomBuffer.contiguous().data_ptr()),
											 reinterpret_cast<char *>(binningBuffer.contiguous().data_ptr()),
											 reinterpret_cast<char *>(imageBuffer.contiguous().data_ptr()),
											 dL_dout_color.contiguous().data<float>(),
											 dL_dout_features.contiguous().data<float>(),
											 dL_dopacity.contiguous().data<float>(),
											 dL_dcolors.contiguous().data<float>(),
											 dL_dmeans3D.contiguous().data<float>(),
											 dL_dsh.contiguous().data<float>(),
											 dL_dscales.contiguous().data<float>(),
											 dL_drotations.contiguous().data<float>(),
											 dL_dtextureFeatures.contiguous().data<float>(),
											 debug);
	}

	return std::make_tuple(dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dsh, dL_dscales, dL_drotations, dL_dtextureFeatures);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansErrorStatsCUDA(
    const torch::Tensor &background,
    const torch::Tensor &means3D,
	const torch::Tensor &opacity,
	const torch::Tensor &scales,
	const torch::Tensor &rotations,
    const torch::Tensor &colors,
    const float tan_fovx,
    const float tan_fovy,
    const torch::Tensor &geometry_buffer,
    const torch::Tensor &texture_buffer,
    const int R,
	const torch::Tensor& viewmatrix,
    const torch::Tensor &binning_buffer,
    const torch::Tensor &image_buffer,
    const torch::Tensor &weight_precomputed,
    const torch::Tensor &loss_img,
    const bool debug
) {
    const int n_gaussians = means3D.size(0);
    const int height      = loss_img.size(1);
    const int width       = loss_img.size(2);

    auto          float_tensor_options = means3D.options();
    auto          int_tensor_options   = float_tensor_options.dtype(torch::kInt32);
    torch::Tensor n_pixels             = torch::zeros({n_gaussians, 1}, int_tensor_options);
    torch::Tensor contributions        = torch::zeros({n_gaussians, 1}, float_tensor_options);
    torch::Tensor errors               = torch::zeros({n_gaussians, 1}, float_tensor_options);
    torch::Tensor view_first_moments   = torch::zeros({n_gaussians, 2}, float_tensor_options);
    torch::Tensor view_second_moments  = torch::zeros({n_gaussians, 3}, float_tensor_options);

    if (n_gaussians == 0) return std::make_tuple(n_pixels, contributions, errors, view_first_moments, view_second_moments);

    ErrorStats::ViewErrorStats view_error_stats = {
        tensor_data_ptr<int32_t>(n_pixels),
        tensor_data_ptr<float>(contributions),
        tensor_data_ptr<float>(errors),
        tensor_data_ptr<float, glm::vec2>(view_first_moments),
        tensor_data_ptr<float, glm::vec3>(view_second_moments)
    };

    CudaRasterizer::Rasterizer::errorStats(
        n_gaussians,
        R,
        tensor_data_ptr<float>(background),
        width,
        height,
		tensor_data_ptr<float>(means3D),
		tensor_data_ptr<float>(opacity),
		tensor_data_ptr<float>(scales),
		tensor_data_ptr<float>(rotations),
        tensor_data_ptr<float>(viewmatrix),
        tan_fovx,
        tan_fovy,
        // No specialization of Tensor::data_ptr for char
        tensor_data_ptr<uint8_t, char>(geometry_buffer),
        tensor_data_ptr<uint8_t, char>(texture_buffer),
        tensor_data_ptr<uint8_t, char>(binning_buffer),
        tensor_data_ptr<uint8_t, char>(image_buffer),
        tensor_data_ptr<float>(weight_precomputed),
        tensor_data_ptr<float>(loss_img),
        view_error_stats,
        debug
    );

    return std::make_tuple(n_pixels, contributions, errors, view_first_moments, view_second_moments);
}

torch::Tensor markVisible(
	torch::Tensor &means3D,
	torch::Tensor &viewmatrix,
	torch::Tensor &projmatrix)
{
	const int P = means3D.size(0);

	torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));

	if (P != 0)
	{
		CudaRasterizer::Rasterizer::markVisible(P,
												means3D.contiguous().data<float>(),
												viewmatrix.contiguous().data<float>(),
												projmatrix.contiguous().data<float>(),
												present.contiguous().data<bool>());
	}

	return present;
}

namespace fix_error
{
	__device__ void computeColorFromSH(const int idx, const int *degs, int max_coeffs, const glm::vec3 *means, glm::vec3 campos, const float *shs, glm::vec3 *out_colours)
	{
		// The implementation is loosely based on code for
		// "Differentiable Point-Based Radiance Fields for
		// Efficient View Synthesis" by Zhang et al. (2022)
		glm::vec3 pos = means[idx];
		glm::vec3 dir = pos - campos;
		dir = dir / glm::length(dir);

		glm::vec3 *sh = ((glm::vec3 *)shs) + idx * max_coeffs;
		glm::vec3 result = SH_C0 * sh[0];
		result += 0.5f;

		const int deg = degs[idx];

		out_colours[idx * 4 + 0] = glm::max(result, 0.0f);
		if (deg == 0)
			return;

		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];
		out_colours[idx * 4 + 1] = glm::max(result, 0.0f);
		if (deg == 1)
			return;

		float xx = x * x, yy = y * y, zz = z * z;
		float xy = x * y, yz = y * z, xz = x * z;
		result = result +
				 SH_C2[0] * xy * sh[4] +
				 SH_C2[1] * yz * sh[5] +
				 SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				 SH_C2[3] * xz * sh[7] +
				 SH_C2[4] * (xx - yy) * sh[8];

		out_colours[idx * 4 + 2] = glm::max(result, 0.0f);
		if (deg == 2)
			return;

		result = result +
				 SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
				 SH_C3[1] * xy * z * sh[10] +
				 SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
				 SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
				 SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
				 SH_C3[5] * z * (xx - yy) * sh[14] +
				 SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
		out_colours[idx * 4 + 3] = glm::max(result, 0.0f);

		return;
	}
};

__global__ void buildRotations(
	const int P,
	const glm::vec4 *rotations,
	float *rotations_matrices)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rotations[idx]; // / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y));

	rotations_matrices[idx * 9 + 0] = R[0][0];
	rotations_matrices[idx * 9 + 1] = R[0][1];
	rotations_matrices[idx * 9 + 2] = R[0][2];
	rotations_matrices[idx * 9 + 3] = R[1][0];
	rotations_matrices[idx * 9 + 4] = R[1][1];
	rotations_matrices[idx * 9 + 5] = R[1][2];
	rotations_matrices[idx * 9 + 6] = R[2][0];
	rotations_matrices[idx * 9 + 7] = R[2][1];
	rotations_matrices[idx * 9 + 8] = R[2][2];
	return;
}

__global__ void transformCentersNdc(
	const int P,
	const glm::vec3 *centers,
	const glm::mat4 *projmatrix,
	const glm::mat4 *inverse_projmatrix,
	const int *image_height,
	const int *image_width,
	const bool aggregateMax,
	float *pixel_sizes)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Transform point by projecting
	glm::vec4 p_orig = glm::vec4(centers[idx], 1.f);
	glm::vec4 p_hom = *projmatrix * p_orig;

	glm::vec3 p_proj =  glm::vec3(normaliseHomogenousPoint(p_hom));
	float depth = p_proj.z;
	bool isInside = glm::all(
		glm::lessThanEqual(glm::vec3(p_proj), glm::vec3(1.)) &&
		glm::greaterThanEqual(glm::vec3(p_proj), glm::vec3(-1, -1, 0)));

	if (isInside)
	{
		glm::vec4 p_proj_end(0.f);
		if (*image_width > *image_height)
			p_proj_end.x = 2.f / *image_width;
		else
			p_proj_end.y = 2.f / *image_height;
		p_proj_end.z = depth;
		p_proj_end.w = 1.0f;

		glm::vec4 p_proj_start(0.f);
		p_proj_start.z = depth;
		p_proj_start.w = 1.0f;

		glm::vec4 p_orig_end = *inverse_projmatrix * p_proj_end;

		glm::vec3 p_orig_end_norm = glm::vec3(normaliseHomogenousPoint(p_orig_end));

		glm::vec4 p_orig_start = *inverse_projmatrix * p_proj_start;
		glm::vec3 p_orig_start_norm = glm::vec3(normaliseHomogenousPoint(p_orig_start));

		glm::vec3 difference = p_orig_end_norm - p_orig_start_norm;
		if (aggregateMax)
		{
			pixel_sizes[idx] = max(pixel_sizes[idx], glm::length(difference));
		}
		else {
			pixel_sizes[idx] = min(pixel_sizes[idx], glm::length(difference));
		}
	}
}

torch::Tensor
calculatePixelSize(
	const torch::Tensor &w2ndc_transforms,
	const torch::Tensor &w2ndc_transforms_inverse,
	const torch::Tensor &means3D,
	const torch::Tensor &image_height,
	const torch::Tensor &image_width,
	const float initial_value,
	const bool aggregateMax)
{
	const int P = means3D.size(0);
	auto float_opts = means3D.options().dtype(torch::kFloat32);

	torch::Tensor pixel_values = torch::full({P, 1}, initial_value, float_opts);

	for (int i = 0; i < w2ndc_transforms.size(0); ++i)
	{
		transformCentersNdc<<<(P + 255) / 256, 256>>>(
			P,
			(glm::vec3 *)means3D.contiguous().data<float>(),
			(glm::mat4 *)w2ndc_transforms.index({i}).contiguous().data<float>(),
			(glm::mat4 *)w2ndc_transforms_inverse.index({i}).contiguous().data<float>(),
			image_height.index({i}).contiguous().data<int>(),
			image_width.index({i}).contiguous().data<int>(),
			aggregateMax,
			pixel_values.contiguous().data<float>());
	}
	return pixel_values;
}

// https://alexminnaar.com/2019/03/05/cuda-kmeans.html
__device__ float distance(const float x1, const float x2)
{
	float dist_sq = (x2 - x1) * (x2 - x1);
	return dist_sq == 0 ? 0 : sqrt(dist_sq);
}

__global__ void updateCenters(
	float *values,
	int *ids,
	float *centers,
	int *center_sizes,
	int n_values,
	int n_centers)
{
	auto idx = cg::this_grid().thread_rank();
	auto block = cg::this_thread_block();

	if (idx >= n_values)
		return;

	int s_idx = threadIdx.x;
	if (s_idx - block.thread_rank() != 0)
	{
		printf("%d, %d\n", s_idx, block.thread_rank());
	}

	__shared__ float collected_values[256];
	collected_values[block.thread_rank()] = values[idx];

	__shared__ int collected_ids[256];
	if (ids[idx] > 255)
	{
		printf("%d\n", ids[idx]);
	}
	collected_ids[block.thread_rank()] = ids[idx];
	// printf("block_idx: %d value %f id %d\n", block.thread_rank(), values[idx], ids[idx]);

	block.sync();

	if (block.thread_rank() == 0)
	{
		// for (int i = 0; i < 256; ++i)
		// {
		// 	printf("i %d clust_id %d\n", i, collected_ids[i]);
		// }
		float block_center_sums[256] = {0};
		int block_center_sizes[256] = {0};
		for (int i = 0; i < 256 && idx + i < n_values; ++i)
		{
			int clust_id = collected_ids[i];
			if (clust_id > 255)
			{
				printf("idx: %d, i: %d clust_id: %d\n", idx, i, clust_id);
			}
			block_center_sums[clust_id] += collected_values[i];
			block_center_sizes[clust_id] += 1;
		}

		for (int i = 0; i < n_centers; ++i)
		{
			atomicAdd(&centers[i], block_center_sums[i]);
			atomicAdd(&center_sizes[i], block_center_sizes[i]);
		}
	}
}

__global__ void updateIds(
	float *values,
	int *ids,
	float *centers,
	int n_values,
	int n_centers)
{
	// get idx for this datapoint
	auto idx = cg::this_grid().thread_rank();
	auto block = cg::this_thread_block();

	// bounds check
	if (idx >= n_values)
		return;

	// find the closest centroid to this datapoint
	float min_dist = INFINITY;
	int closest_centroid = 0;

	__shared__ float collected_centers[256];

	block.sync();
	collected_centers[block.thread_rank()] = centers[block.thread_rank()];
	block.sync();

	for (int i = 0; i < n_centers; ++i)
	{
		float dist = distance(values[idx], collected_centers[i]);

		if (dist < min_dist)
		{
			min_dist = dist;
			closest_centroid = i;
		}
	}

	// assign closest cluster id for this datapoint/thread
	if (closest_centroid >= 256)
	{
		printf("closest_centroid %d", closest_centroid);
	}
	ids[idx] = closest_centroid;
}

void updateCentersWrapper(
	float *values,
	int *ids,
	float *centers,
	int *center_sizes,
	int n_values,
	int n_centers)
{
	updateCenters<<<(n_values + 255) / 256, 256>>>(
		values,
		ids,
		centers,
		center_sizes,
		n_values,
		n_centers);
}

void updateIdsWrapper(
	float *values,
	int *ids,
	float *centers,
	int n_values,
	int n_centers)
{
	updateIds<<<(n_values + 255) / 256, 256>>>(
		values,
		ids,
		centers,
		n_values,
		n_centers);
}

// Works with 256 centers 1 dimensional data only
std::tuple<torch::Tensor, torch::Tensor>
kmeans(
	const torch::Tensor &values,
	const torch::Tensor &centers,
	const float tol,
	const int max_iterations)
{
	const int n_values = values.size(0);
	const int n_centers = centers.size(0);
	torch::Tensor ids = torch::zeros({n_values, 1}, values.options().dtype(torch::kInt32));
	torch::Tensor new_centers = torch::zeros({n_centers}, values.options().dtype(torch::kFloat32));
	torch::Tensor old_centers = torch::zeros({n_centers}, values.options().dtype(torch::kFloat32));
	new_centers = centers.clone();
	torch::Tensor center_sizes = torch::zeros({n_centers}, values.options().dtype(torch::kInt32));

	for (int i = 0; i < max_iterations; ++i)
	{
		CHECK_CUDA(updateIdsWrapper(
					   values.contiguous().data<float>(),
					   ids.contiguous().data<int>(),
					   new_centers.contiguous().data<float>(),
					   n_values,
					   n_centers),
				   false)

		// // // cudaDeviceSynchronize();
		old_centers = new_centers.clone();
		new_centers.zero_();
		CHECK_CUDA(, false)
		center_sizes.zero_();
		CHECK_CUDA(, false)

		CHECK_CUDA(updateCentersWrapper(
					   values.contiguous().data<float>(),
					   ids.contiguous().data<int>(),
					   new_centers.contiguous().data<float>(),
					   center_sizes.contiguous().data<int>(),
					   n_values,
					   n_centers),
				   false)

		new_centers = new_centers / center_sizes;
		new_centers.index_put_({new_centers.isnan()}, 0.f);
		// 	if (idx < n_centers)
		// 	{
		// 		const int center_size = center_sizes[idx];
		// 		if (center_size == 0) centers[idx] = 0;
		// 		else centers[idx] = centers[idx] / center_size;
		// 	}
		// if (idx == 0)
		// 	printf("%d %f %d\n", idx, centers[idx], center_sizes[idx]);
		float center_shift = (old_centers - new_centers).abs().sum().item<float>();
		if (center_shift < tol)
			break;
	}

	CHECK_CUDA(updateIdsWrapper(
				   values.contiguous().data<float>(),
				   ids.contiguous().data<int>(),
				   new_centers.contiguous().data<float>(),
				   n_values,
				   n_centers),
			   false)

	return std::make_tuple(ids, new_centers);
}

// Function that handles texture map assignment and resize
void ResizeJaggedTensor(
	torch::Tensor &source_jagged_tensor,
	torch::Tensor &source_resolution,
	torch::Tensor &source_offset,
	torch::Tensor &source_mask,
	torch::Tensor &target_mask,
	torch::Tensor &target_jagged_tensor,
	torch::Tensor &target_resolution,
	torch::Tensor &target_offset,
	torch::Tensor &target_center_shift
)
{
	const int n_iterations = source_mask.size(0);
	if (n_iterations == 0)
	{
		std::cout << "Found 0 primitives to work with. Returning";
		return;
	}

	if (target_mask.options().dtype() == source_mask.options().dtype())
	{
		if (target_mask.options().dtype() == torch::Dtype::Int)
		{
			if (target_mask.size(0) != source_mask.size(0) && target_mask.size(0) != target_center_shift.size(0))
			{
				std::cout << "The provided int masks don't have the same size. Returning";
				return;
			}
		}
	}
	else
	{
		std::cout << "The provided masks don't have the same type. Returning";
		return;
	}

	UTILS::CopyTensor(
        n_iterations,
        source_jagged_tensor.contiguous().data<float>(),
        (int2 *)source_resolution.contiguous().data<int>(),
        source_offset.contiguous().data<int>(),
		source_mask.contiguous().data<int>(),
		target_mask.contiguous().data<int>(),
        target_jagged_tensor.contiguous().data<float>(),
        (int2 *)target_resolution.contiguous().data<int>(),
        target_offset.contiguous().data<int>(),
        (int2 *)target_center_shift.contiguous().data<int>()
	);
	return;
}

void CreateJaggedMask(
	torch::Tensor &mask,
	torch::Tensor &jagged_tensor,
	torch::Tensor &resolution,
	torch::Tensor &offset
)
{
	const int n_primitives = mask.size(0);
	UTILS::CreateJaggedMask(
        n_primitives,
		mask.contiguous().data<bool>(),
        jagged_tensor.contiguous().data<bool>(),
        (int2 *)resolution.contiguous().data<int>(),
        offset.contiguous().data<int>()
	);
}


