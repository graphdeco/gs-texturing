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

#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
//#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"
#include "error_stats.h"


// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool* present)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;
	present[idx] = in_frustum(idx, orig_points, viewmatrix, false, p_view);
}

// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(
	int P,
	const float2* points_xy,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	int* radii,
	dim3 grid)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible Gaussians
	if (radii[idx] > 0)
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint2 rect_min, rect_max;

		getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

		// For each tile that the bounding rect overlaps, emit a 
		// key/value pair. The key is |  tile ID  |      depth      |,
		// and the value is the ID of the Gaussian. Sorting the values 
		// with this key yields Gaussian IDs in a list, such that they
		// are first sorted by tile and then by depth. 
		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				uint64_t key = y * grid.x + x;
				key <<= 32;
				key |= *((uint32_t*)&depths[idx]);
				gaussian_keys_unsorted[off] = key;
				gaussian_values_unsorted[off] = idx;
				off++;
			}
		}
	}
}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}

// Mark Gaussians as visible/invisible, based on view frustum testing
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present)
{
	checkFrustum << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}

CudaRasterizer::TextureGradients::TextureGradients(float *dL_dtextureMap) : dL_dtextureMap(dL_dtextureMap)
{
}

CudaRasterizer::TextureState::TextureState(){}

CudaRasterizer::TextureState::TextureState(const glm::vec3 *textureMap,
										   const int2 *textureResolution,
										   const int *textureMapStartingOffset,
										   const float *texelSize) : textureMap(textureMap),
																	 textureResolution(textureResolution),
																	 textureMapStartingOffset(textureMapStartingOffset),
																	 texelSize(texelSize)
{
}

CudaRasterizer::TextureState::Buffer CudaRasterizer::TextureState::Buffer::fromChunk(char *&chunk, size_t P)
{
	TextureState::Buffer buffer;
	obtain(chunk, buffer.view2canonical, P, 128);
	obtain(chunk, buffer.normal, P, 128);
	obtain(chunk, buffer.normal_sign, P, 128);
	obtain(chunk, buffer.mean, P, 128);
	return buffer;
}

CudaRasterizer::GeometryState::Buffer CudaRasterizer::GeometryState::Buffer::fromChunk(char*& chunk, size_t P)
{
	GeometryState::Buffer buffer;
	obtain(chunk, buffer.depths, P, 128);
	obtain(chunk, buffer.clamped, P * 3, 128);
	obtain(chunk, buffer.internal_radii, P, 128);
	obtain(chunk, buffer.means2D, P, 128);
	obtain(chunk, buffer.transMat, P * 9, 128);
	obtain(chunk, buffer.conic_opacity, P, 128);
	obtain(chunk, buffer.rgb, P * 3, 128);
	obtain(chunk, buffer.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, buffer.scan_size, buffer.tiles_touched, buffer.tiles_touched, P);
	obtain(chunk, buffer.scanning_space, buffer.scan_size, 128);
	obtain(chunk, buffer.point_offsets, P, 128);
	return buffer;
}

CudaRasterizer::GeometryState::GeometryState(){}
CudaRasterizer::GeometryState::GeometryState(const glm::vec2 *scale,
											 const glm::vec3 *mean,
											 const glm::vec4 *rotation,
											 const float *opacity) : scale(scale),
																	 mean(mean),
																	 rotation(rotation),
																	 opacity(opacity)
{
}

CudaRasterizer::GeometryGradients::GeometryGradients(
	const float *dL_dpixels,
	const float *dL_dout_features,
	glm::vec2 *dL_dscale,
	glm::vec4 *dL_dquaternion,
	glm::vec3 *dL_dmean3D,
	float *dL_dopacity) : dL_dpixels(dL_dpixels),
						 dL_dout_features(dL_dout_features),
						 dL_dscale(dL_dscale),
						 dL_dquaternion(dL_dquaternion),
						 dL_dmean3D(dL_dmean3D),
						 dL_dopacity(dL_dopacity)
{
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.accum_alpha, N, 128);
	obtain(chunk, img.n_contrib, N, 128);
	obtain(chunk, img.ranges, N, 128);
	return img;
}

CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
int CudaRasterizer::Rasterizer::forward(
	std::function<char *(size_t)> textureBuffer,
	std::function<char *(size_t)> geometryBuffer,
	std::function<char *(size_t)> binningBuffer,
	std::function<char *(size_t)> imageBuffer,
	const int P, int D, int M,
	const float *background,
	const int width, int height,
	const float *means3D,
	const float *shs,
	const float3 *textureMap,
	const int2 *textureResolution,
	const int *textureMapStartingOffset,
	const float *texelSize,
	const float *colors_precomp,
	const float *opacities,
	const float *scales,
	const float *rotations,
	const float *transMat_precomp,
	const float *viewmatrix,
	const float *projmatrix,
	const float *cam_pos,
	const float2 tan_fov,
	const bool prefiltered,
	float *out_color,
	float *out_features,
	int *out_touched_pixels,
	float *out_transmittance,
	const ColourType colourType,
	int *radii,
	const bool calculate_mean_transmittance,
	const bool debug,
	const bool texture_debug_view)
{
	const float2 focal{width / (2.0f * tan_fov.x), height / (2.0f * tan_fov.y)};

	TextureState textureState((const glm::vec3 *)textureMap, textureResolution, textureMapStartingOffset, texelSize);
	size_t chunk_size = required<TextureState::Buffer>(P);
	char* texture_chunkptr = textureBuffer(chunk_size);
	textureState.buffer = TextureState::Buffer::fromChunk(texture_chunkptr, P);

	GeometryState geomState((const glm::vec2 *)scales, (const glm::vec3 *)means3D, (const glm::vec4 *)rotations, (const float *)opacities);
	chunk_size = required<GeometryState::Buffer>(P);
	char* chunkptr = geometryBuffer(chunk_size);
	geomState.buffer = GeometryState::Buffer::fromChunk(chunkptr, P);

	if (radii == nullptr)
    {
		radii = geomState.buffer.internal_radii;
    }

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Dynamically resize image-based auxiliary buffers during training
	size_t img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	CHECK_CUDA(FORWARD::preprocess(
		P, D, M,
		shs,
		transMat_precomp,
		colors_precomp,
		viewmatrix, projmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		focal,
		tan_fov,
		radii,
		textureState,
		geomState,
		tile_grid,
		prefiltered,
		colourType
	), debug)

	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.buffer.scanning_space, geomState.buffer.scan_size,
	geomState.buffer.tiles_touched, geomState.buffer.point_offsets, P), debug)

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	int num_rendered;
	CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.buffer.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated Gaussian indices to be sorted
	duplicateWithKeys << <(P + 255) / 256, 256 >> > (
		P,
		geomState.buffer.means2D,
		geomState.buffer.depths,
		geomState.buffer.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radii,
		tile_grid)
		CHECK_CUDA(, debug)

	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	// Sort complete list of (duplicated) Gaussian indices by keys
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit), debug)

	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

	// Identify start and end of per-tile workloads in sorted list
	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges);
	CHECK_CUDA(, debug)

	// Let each tile blend its range of Gaussians independently in parallel
	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.buffer.rgb;
	CHECK_CUDA(FORWARD::render(
		tile_grid, block,
		binningState.point_list,
		width, height,
		textureState,
		geomState,
		imgState,
		focal,
		feature_ptr,
		background,
		colourType,
		viewmatrix,
		out_color,
		out_features,
		out_touched_pixels,
		out_transmittance,
		calculate_mean_transmittance,
		texture_debug_view), debug)

	return num_rendered;
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::backward(
	const int P, int D, int M, int R,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float3 *textureMap,
	const int2 *textureResolution,
	const int *textureMapStartingOffset,
	const float *texelSize,
	const float* colors_precomp,
	const float* scales,
	const float* opacities,
	const float* rotations,
	const float* transMat_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* campos,
	const float2 tan_fov,
	const int* radii,
	char* texture_buffer,
	char* geom_buffer,
	char* binning_buffer,
	char* img_buffer,
	const float* dL_dpix,
	const float* dL_dout_features,
	float* dL_dopacity,
	float* dL_dcolor,
	float* dL_dmean3D,
	float* dL_dsh,
	float* dL_dscale,
	float* dL_drot,
	float* dL_dtextureFeatures,
	bool debug)
{
	TextureState textureState((const glm::vec3 *)textureMap, textureResolution, textureMapStartingOffset, texelSize);
	textureState.buffer = TextureState::Buffer::fromChunk(texture_buffer, P);
	GeometryState geomState((const glm::vec2 *)scales, (const glm::vec3 *)means3D, (const glm::vec4 *)rotations, (const float *)opacities);
	geomState.buffer = GeometryState::Buffer::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height);

	// Initialise Gradients objects
	TextureGradients textureGrads(dL_dtextureFeatures);
    GeometryGradients geomGrads(
		dL_dpix,
		dL_dout_features,
		(glm::vec2 *)dL_dscale,
		(glm::vec4 *)dL_drot,
		(glm::vec3 *)dL_dmean3D,
		dL_dopacity
	);


	if (radii == nullptr)
	{
		radii = geomState.buffer.internal_radii;
	}

	const float2 focal = make_float2(width / (2.0f * tan_fov.x), height / (2.0f * tan_fov.y));

	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.buffer.rgb;
	CHECK_CUDA(BACKWARD::render(
				   P,
				   tile_grid,
				   block,
				   binningState.point_list,
				   width, height,
				   background,
				   textureState,
				   geomState,
				   imgState,
				   textureGrads,
				   geomGrads,
				   color_ptr,
				   tan_fov,
				   focal,
				   viewmatrix,
				   dL_dcolor),
			   debug)

	// Take care of the rest of preprocessing. Was the precomputed covariance
	// given to us or a scales/rot pair? If precomputed, pass that. If not,
	// use the one we computed ourselves.
	CHECK_CUDA(BACKWARD::preprocess(P, D, M,
		radii,
		shs,
		geomState,
		geomGrads,
		viewmatrix,
		projmatrix,
		focal,
		tan_fov,
		(glm::vec3*)campos,
		dL_dcolor,
		dL_dsh), debug)
}

void CudaRasterizer::Rasterizer::errorStats(
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
) {
	GeometryState geometry_state((const glm::vec2 *)scales, (const glm::vec3 *)means3D, (const glm::vec4 *)rotations, (const float *)opacity);
	geometry_state.buffer = GeometryState::Buffer::fromChunk(const_cast<char *&>(geometry_buffer), n_gaussians);

	TextureState texture_state;
	texture_state.buffer = TextureState::Buffer::fromChunk(const_cast<char *&>(texture_buffer), n_gaussians);

    BinningState  binning_state  = BinningState::fromChunk(const_cast<char *&>(binning_buffer), R);
    ImageState    image_state    = ImageState::fromChunk(const_cast<char *&>(image_buffer), width * height);

    const float focal_x = width / (2.0f * tan_fovx);
    const float focal_y = height / (2.0f * tan_fovy);

    const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
    const dim3 block(BLOCK_X, BLOCK_Y, 1);

    const float *color_ptr = geometry_state.buffer.rgb;
    CHECK_CUDA(
        ErrorStats::render(
            tile_grid,
            block,
            image_state.ranges,
            binning_state.point_list,
            width,
            height,
			make_float2(focal_x, focal_y),
			viewmatrix,
            background,
			geometry_state,
			texture_state,
			weight_precomputed,
            image_state.accum_alpha,
            image_state.n_contrib,
            loss_img,
            view_error_stats
        ),
        debug
    )

    CHECK_CUDA(
        ErrorStats::remove_pixel_scaling(
            focal_x,
            focal_y,
            view_error_stats.first_moments,
            view_error_stats.second_moments,
            n_gaussians
        ),
        debug
    )
}