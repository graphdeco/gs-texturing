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

#ifndef CUDA_RASTERIZER_BACKWARD_H_INCLUDED
#define CUDA_RASTERIZER_BACKWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include "rasterizer_impl.h"

namespace BACKWARD
{
	void render(
		const int n_primitives,
		const dim3 grid, const dim3 block,
		const uint32_t *point_list,
		int W, int H,
		const float *bg_color,
		const CudaRasterizer::TextureState& textureState,
		const CudaRasterizer::GeometryState& geomState,
		const CudaRasterizer::ImageState& imgState,
		const CudaRasterizer::TextureGradients &textureGrads,
		const CudaRasterizer::GeometryGradients &geomGrads,
		const float *colors,
		const float2 tan_fov,
		const float2 focal,
		const float *viewmatrix,
		float *dL_dcolors);

	void preprocess(
		int P, const int D, int M,
		const int* radii,
		const float* shs,
		const CudaRasterizer::GeometryState& geomState,
		const CudaRasterizer::GeometryGradients &geomGrads,
		const float* view,
		const float* proj,
		const float2 focal,
		const float2 tan_fov,
		const glm::vec3* campos,
		float* dL_dcolor,
		float* dL_dsh);
}

#endif