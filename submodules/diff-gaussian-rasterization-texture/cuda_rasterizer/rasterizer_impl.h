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

#include <iostream>
#include <vector>
#include "rasterizer.h"
#include <cuda_runtime_api.h>
// #define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace CudaRasterizer
{
	enum ColourType : unsigned int
	{
		FULL = 0,
		BASE = 1,
		HIGHLIGHTS = 2
	};

	template <typename T>
	static void obtain(char *&chunk, T *&ptr, std::size_t count, std::size_t alignment)
	{
		std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
		ptr = reinterpret_cast<T *>(offset);
		chunk = reinterpret_cast<char *>(ptr + count);
	}

	struct TextureState
	{
		struct Buffer
		{
			glm::mat3 *view2canonical;
			glm::vec3 *normal;
			int *normal_sign;
			float3 *mean;
			static Buffer fromChunk(char *&chunk, size_t P);
		};

		Buffer buffer;
		const glm::vec3 *textureMap;
		const int2 *textureResolution;
		const int *textureMapStartingOffset;
		const float *texelSize;

		TextureState();
		TextureState(const glm::vec3 *textureMap,
					 const int2 *textureResolution,
					 const int *textureMapStartingOffset,
					 const float *texelSize);
	};

	struct TextureGradients
	{
		float *dL_dtextureMap;

		TextureGradients(float *dL_dtextureMap);
	};

	struct GeometryState
	{
		struct Buffer
		{
			size_t scan_size;
			float *depths;
			char *scanning_space;
			bool *clamped;
			int *internal_radii;
			float2 *means2D;
			float *transMat;
			float4 *conic_opacity;
			float *rgb;
			uint32_t *point_offsets;
			uint32_t *tiles_touched;
			static Buffer fromChunk(char *&chunk, size_t P);
		};

		Buffer buffer;
		const glm::vec2 *scale = nullptr;
		const glm::vec3 *mean = nullptr;
		const glm::vec4 *rotation = nullptr;
		const float *opacity = nullptr;

		GeometryState();
		GeometryState(const glm::vec2 *scale,
					  const glm::vec3 *mean,
					  const glm::vec4 *rotation,
					  const float *opacity);
	};

	struct GeometryGradients
	{
		const float *dL_dpixels;
		const float *dL_dout_features;
		glm::vec2 *dL_dscale;
		glm::vec4 *dL_dquaternion;
		glm::vec3 *dL_dmean3D;
		float *dL_dopacity;

		GeometryGradients(
			const float *dL_dpixels,
			const float *dL_dout_features,
			glm::vec2 *dL_dscale,
			glm::vec4 *dL_dquaternion,
			glm::vec3 *dL_dmean3D,
			float *dL_dopacity);
	};

	struct ImageState
	{
		uint2 *ranges;
		uint32_t *n_contrib;
		float *accum_alpha;

		static ImageState fromChunk(char *&chunk, size_t N);
	};

	struct BinningState
	{
		size_t sorting_size;
		uint64_t *point_list_keys_unsorted;
		uint64_t *point_list_keys;
		uint32_t *point_list_unsorted;
		uint32_t *point_list;
		char *list_sorting_space;

		static BinningState fromChunk(char *&chunk, size_t P);
	};

	template <typename T>
	size_t required(size_t P)
	{
		char *size = nullptr;
		T::fromChunk(size, P);
		return ((size_t)size) + 128;
	}
};