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

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include "rasterizer_impl.h"

namespace FORWARD
{
	// Perform initial steps for each Gaussian prior to rasterization.
	void preprocess(int P, const int D, int M,
		const float* shs,
		const float* transMat_precomp,
		const float* colors_precomp,
		const float* viewmatrix,
		const float* projmatrix,
		const glm::vec3* cam_pos,
		const int W, int H,
		const float2 focal,
		const float2 tan_fov,
		int* radii,
		CudaRasterizer::TextureState& textureState,
		CudaRasterizer::GeometryState& geomState,
		const dim3 grid,
		bool prefiltered,
		const CudaRasterizer::ColourType colourType);

	// Main rasterization method.
	void render(
		const dim3 grid, dim3 block,
		const uint32_t* point_list,
		int W, int H,
		const CudaRasterizer::TextureState& textureState,
		const CudaRasterizer::GeometryState& geomState,
		const CudaRasterizer::ImageState& imgState,
		const float2 focal,
		const float* features,
		const float* bg_color,
		const CudaRasterizer::ColourType colourType,
		const float *viewmatrix,
		float* out_color,
		float* out_features,
		int* out_touched_pixels,
		float* out_transmittance,
		const bool calculate_mean_transmittance = false,
		const bool texture_debug_view = false);
}


#endif