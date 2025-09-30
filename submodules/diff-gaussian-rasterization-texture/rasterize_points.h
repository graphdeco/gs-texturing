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
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor &background,
	const torch::Tensor &means3D,
	const torch::Tensor &colors,
	const torch::Tensor &opacity,
	const torch::Tensor &scales,
	const torch::Tensor &rotations,
	const torch::Tensor &transMat_precomp,
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
	const bool calculate_mean_weight = false,
	const bool debug = false,
	const bool texture_debug_view = false);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
 	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
    const torch::Tensor& colors,
	const torch::Tensor& scales,
	const torch::Tensor& opacities,
	const torch::Tensor& rotations,
	const torch::Tensor& transMat_precomp,
	const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
	const torch::Tensor &textureMap,
	const torch::Tensor &textureResolution,
	const torch::Tensor &textureMapStartingOffset,
	const torch::Tensor &texelSize,
    const torch::Tensor& dL_dout_color,
    const torch::Tensor& dL_dout_features,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& textureBuffer,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool debug = false);

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
	const torch::Tensor &viewmatrix,
    const torch::Tensor &binning_buffer,
    const torch::Tensor &image_buffer,
    const torch::Tensor &weight_precomputed,
    const torch::Tensor &loss_img,
    const bool debug
);

torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
calculateColourVariance(
	const torch::Tensor &cam_positions,
	const torch::Tensor &background,
	const torch::Tensor &means3D,
	const torch::Tensor &colors,
	const torch::Tensor &opacity,
	const torch::Tensor &scales,
	const torch::Tensor &rotations,
	const torch::Tensor &cov3D_precomp,
	const torch::Tensor &cam_viewmatrices,
	const torch::Tensor &cam_projmatrices,
	const torch::Tensor tan_fovxs,
	const torch::Tensor tan_fovys,
	const torch::Tensor image_height,
	const torch::Tensor image_width,
	const torch::Tensor &sh,
	const torch::Tensor &textureMap,
	const torch::Tensor &textureResolution,
	const torch::Tensor &textureMapStartingOffset,
	const torch::Tensor &texelSize,
	const torch::Tensor &degrees,
	const int deg);

torch::Tensor
calculatePixelSize(
	const torch::Tensor &w2ndc_transforms,
	const torch::Tensor &w2ndc_transforms_inverse,
	const torch::Tensor &means3D,
	const torch::Tensor &image_height,
	const torch::Tensor &image_width,
	const float initial_value = 10000,
	const bool aggregateMax = false);

std::tuple<torch::Tensor, torch::Tensor>
kmeans(
	const torch::Tensor &values,
	const torch::Tensor &centers,
	const float tol,
	const int max_iterations
);

void ResizeJaggedTensor(
	torch::Tensor &source_jagged_tensor,
	torch::Tensor &source_resolution,
	torch::Tensor &source_offset,
	torch::Tensor &source_mask,
	torch::Tensor &target_mask,
	torch::Tensor &target_jagged_tensor,
	torch::Tensor &target_resolution,
	torch::Tensor &target_offset,
	torch::Tensor &target_ceneter_shift
);

void CreateJaggedMask(
	torch::Tensor &mask,
	torch::Tensor &jagged_tensor,
	torch::Tensor &resolution,
	torch::Tensor &offset
);
