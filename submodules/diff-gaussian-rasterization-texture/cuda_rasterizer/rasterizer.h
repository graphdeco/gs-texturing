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

#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>
#include "error_stats_types.h"

namespace CudaRasterizer
{
	enum ColourType: unsigned int;
	class Rasterizer
	{
	public:

		static void markVisible(
			int P,
			float* means3D,
			float* viewmatrix,
			float* projmatrix,
			bool* present);

		static int forward(
			std::function<char* (size_t)> textureBuffer,
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int P, const int D, int M,
			const float* background,
			const int width, const int height,
			const float* means3D,
			const float* shs,
			const float3* textureMap,
			const int2 *textureResolution,
			const int *textureMapStartingOffset,
			const float* texelSize,
			const float* colors_precomp,
			const float* opacities,
			const float* scales,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* cam_pos,
			const float2 tan_fov,
			const bool prefiltered,
			float* out_color,
			float* out_features,
			int* out_touched_pixels,
			float* out_transmittance,
			const CudaRasterizer::ColourType colourType,
			int* radii = nullptr,
			const bool calculate_mean_transmittance = false,
			const bool debug = false,
			const bool texture_debug_view = false);

		static void backward(
			const int P, const int D, int M, int R,
			const float* background,
			const int width, int height,
			const float* means3D,
			const float* shs,
			const float3* textureMap,
			const int2 *textureResolution,
			const int *textureMapStartingOffset,
			const float* texelSize,
			const float* colors_precomp,
			const float* scales,
			const float* opacities,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* campos,
			const float2 tan_fov,
			const int* radii,
			char* texture_buffer,
			char* geom_buffer,
			char* binning_buffer,
			char* image_buffer,
			const float* dL_dpix,
			const float* dL_dout_normal,
			float* dL_dopacity,
			float* dL_dcolor,
			float* dL_dmean3D,
			float* dL_dsh,
			float* dL_dscale,
			float* dL_drot,
			float* dL_dtextureFeatures,
			bool debug = false);

		static int inferenceForward(
			std::function<char* (size_t)> textureBuffer,
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int P, int* D, const int bandsNum, const int* coeffsNum,
			const int* perBandPrimitiveCount,
			const int* cumSumPrimitiveCount,
			const float* background,
			const int width, int height,
			const float* means3D,
			const float* shs,
			const float3* textureMap,
			const float* colors_precomp,
			const float* opacities,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* cam_pos,
			const float2 tan_fov,
			const bool prefiltered,
			float* out_color,
			int* out_touched_pixels,
			float* out_transmittance,
			int* radii = nullptr,
			const bool calculate_mean_transmittance = false,
			bool debug = false);
		
		static void errorStats(
					int                        n_gaussians,
					int                        R,
					const float               *background,
					int                        width,
					int                        height,
					const float				  *means3D,
					const float				  *opacity,
					const float				  *scales,
					const float				  *rotations,
        			const float				  *viewmatrix,
					float                      tan_fovx,
					float                      tan_fovy,
					const char                *geometry_buffer,
					const char                *texture_buffer,
					const char                *binning_buffer,
					const char                *image_buffer,
                  	const float               *weight_precomputed,
                  	const float               *loss_img,
                    ErrorStats::ViewErrorStats view_error_stats,
					bool                       debug
		);
	};
};

#endif