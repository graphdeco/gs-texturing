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

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <glm/gtc/type_ptr.hpp>
namespace cg = cooperative_groups;

// Returns the offset of the SH tensor when having multiple, different sized subtensors concatenated into one
__device__ int getSHOffset(const int idx, const int* coeffsNum, const int* perBandPrimitiveCount, const int* cumSumPrimitiveCount, int& deg)
{
	int offset = 0;
	deg = 0;
	if (idx < cumSumPrimitiveCount[0]) return idx * coeffsNum[0];
	
	deg = 1;
	offset += perBandPrimitiveCount[0] * coeffsNum[0];
	if (idx < cumSumPrimitiveCount[1]) return offset + (idx - cumSumPrimitiveCount[0]) * coeffsNum[1];
	
	deg = 2;
	offset += perBandPrimitiveCount[1] * coeffsNum[1];
	if (idx < cumSumPrimitiveCount[2]) return offset + (idx - cumSumPrimitiveCount[1]) * coeffsNum[2];
	
	deg = 3;
	offset += perBandPrimitiveCount[2] * coeffsNum[2];
	return offset + (idx - cumSumPrimitiveCount[2]) * coeffsNum[3];
}

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
// Used when having a variable number of SH bands
__device__ glm::vec3 computeColorFromSH(
	const int idx,
	const int* coeffsNum,
	const int* perBandPrimitiveCount,
	const int* cumSumPrimitiveCount,
	const glm::vec3* means,
	glm::vec3 campos,
	const float* shs,
	bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	int deg;
	glm::vec3* sh = ((glm::vec3*)shs) + getSHOffset(idx, coeffsNum, perBandPrimitiveCount, cumSumPrimitiveCount, deg);
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(const int idx,
										const int deg,
										const int max_coeffs,
										const glm::vec3 *means,
										glm::vec3 campos,
										const float *shs,
										bool *clamped,
										const CudaRasterizer::ColourType colourType)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result(0.f);
	if (colourType != CudaRasterizer::ColourType::HIGHLIGHTS)
	{
		result += 0.5f;
		result += SH_C0 * sh[0];
	}

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
__device__ void computeCov2D(const float *cov3D,
							 const glm::mat3 &T,
							 float3 &cov2D)
{

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	// Transpose of the symmetric Vrk is the same matrix
	glm::mat3 cov = T * Vrk * glm::transpose(T);

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	cov2D.x = float(cov[0][0]);
	cov2D.y = float(cov[0][1]);
	cov2D.z = float(cov[1][1]);
}

__device__ void compute_canonical2world(const glm::vec2& scale,
									 const glm::vec4 rot,
									 glm::mat3 &canonical2world,
									 glm::vec3 &normal)
{
	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot; // / glm::length(rot);
	// Compute rotation matrix from quaternion
	glm::mat3 R = buildRotationMatrix(q);
	normal = R[2];

	// Create scaling matrix
	glm::mat3 S = scale_to_mat(scale);

	canonical2world = R * S;
}

// Compute the canonical2world transformation and then transforms it to view space
// by multiplying with the view matrix
__device__ void compute_canonical2view(const glm::vec2 &scale,
									   const glm::vec4 rot,
									   const float3 &mean,
									   const glm::vec3 &p_view,
									   const float *viewmatrix,
									   glm::mat3 &canonical2view,
									   glm::vec3 &normal,
									   int &sign)
{
	glm::mat3 world2view = glm::mat3(
		viewmatrix[0], viewmatrix[1], viewmatrix[2],
		viewmatrix[4], viewmatrix[5], viewmatrix[6],
		viewmatrix[8], viewmatrix[9], viewmatrix[10]);

	glm::mat3 canonical2world(1.f);
	compute_canonical2world(scale, rot, canonical2world, normal);

	canonical2view = world2view * canonical2world;
	normal = world2view * normal;
#ifdef DUAL_VISIBLE
	float cos = -glm::dot(p_view, canonical2view[2]);
	if (cos == 0) return;
	sign = cos > 0 ? 1: -1;
	canonical2view[2] *= (float)sign;
#endif
	normal *= (float)sign;
}

__device__ float computeTextureVariance(
	const glm::vec3 *textureMap,
	const int2 texRes,
	const int2 maxTexSpan)
{
	const int n = texRes.x * texRes.y;
	glm::vec3 sum(0.f);
	glm::vec3 sumSq(0.f);
	const glm::vec<2, int> indexOffset = glm::vec<2, int>(maxTexSpan.x - texRes.x, maxTexSpan.y - texRes.y) / 2;
	for (int i{indexOffset.x}; i < indexOffset.x + texRes.x; ++i)
	{
		for (int j{indexOffset.y}; j < indexOffset.y + texRes.y; ++j)
		{
			const glm::vec3 &texValue = textureMap[j * maxTexSpan.x + i];
			sum += texValue;
			sumSq += texValue * texValue;
		}
	}
	glm::vec3 variance = (sumSq - (sum * sum) / glm::vec3(n)) / glm::vec3(n - 1);
	return (variance[0] + variance[1] + variance[2]) / 3;
}

// Computing the bounding box of the 2D Gaussian and its center
// The center of the bounding box is used to create a low pass filter
__device__ bool compute_aabb(
	const glm::mat3 T,
	glm::vec2 &point_image,
	glm::vec2 &extent)
{
	glm::vec3 T0(T[0][0], T[0][1], T[0][2]);
	glm::vec3 T1(T[1][0], T[1][1], T[1][2]);
	glm::vec3 T3(T[2][0], T[2][1], T[2][2]);

	// Compute AABB
	glm::vec3 temp_point(1.0f, 1.0f, -1.0f);
	float distance = glm::dot(T3 * T3, temp_point);
	glm::vec3 f = (1.f / distance) * temp_point;
	if (distance == 0.0f) return false;

	point_image = {
		glm::dot(f * T0, T3),
		glm::dot(f * T1, T3)
	};  
	
	glm::vec2 temp = {
		glm::dot(f * T0, T0),
		glm::dot(f * T1, T1)
	};
	glm::vec2 half_extend = point_image * point_image - temp;
	extent = glm::sqrt(glm::max(glm::vec2(1e-4f, 1e-4f), half_extend));
	return true;
}

// Compute a 2D-to-2D mapping matrix from a tangent plane into a image plane
// given a 2D gaussian parameters.
__device__ void compute_transmat(
	const float3& p_orig,
	const glm::vec2 scale,
	const glm::vec4 rot,
	const float* projmatrix,
	const float* viewmatrix,
	const int W,
	const int H, 
	glm::mat3 &T
) {

	glm::mat3 R = buildRotationMatrix(rot);
	glm::mat3 S = scale_to_mat(scale);
	glm::mat3 L = R * S;

	// center of Gaussians in the camera coordinate
	glm::mat3x4 splat2world = glm::mat3x4(
		glm::vec4(L[0], 0.0),
		glm::vec4(L[1], 0.0),
		glm::vec4(p_orig.x, p_orig.y, p_orig.z, 1)
	);

	glm::mat4 world2ndc = glm::mat4(
		projmatrix[0], projmatrix[4], projmatrix[8], projmatrix[12],
		projmatrix[1], projmatrix[5], projmatrix[9], projmatrix[13],
		projmatrix[2], projmatrix[6], projmatrix[10], projmatrix[14],
		projmatrix[3], projmatrix[7], projmatrix[11], projmatrix[15]
	);

	glm::mat3x4 ndc2pix = glm::mat3x4(
		glm::vec4(float(W) / 2.0f, 0.0, 0.0, float(W-1) / 2.0f),
		glm::vec4(0.0, float(H) / 2.0f, 0.0, float(H-1) / 2.0f),
		glm::vec4(0.0, 0.0, 0.0, 1.0)
	);

	T = glm::transpose(splat2world) * world2ndc * ndc2pix;
}

// Perform initial steps for each Gaussian prior to rasterization.
template <int C>
__global__ void preprocessCUDA(int P, const int D, int M,
							   const float *orig_points,
							   const glm::vec2 *scales,
							   const glm::vec4 *rotations,
							   const float *opacities,
							   const float *shs,
							   const float *transMat_precomp,
							   const float *colors_precomp,
							   const float *viewmatrix,
							   const float *projmatrix,
							   const glm::vec3 *cam_pos,
							   const int W, int H,
							   const float2 tan_fov,
							   const float2 focal,
							   int *radii,
							   bool *clamped,
							   float2 *points_xy_image,
							   float *depths,
							   float *transMats,
							   float *rgb,
							   float4 *conic_opacity,
							   uint32_t *tiles_touched,
							   float3 *p_view,
							   glm::mat3 *view2canonical,
							   glm::vec3 *normal,
							   int *normal_sign,
							   const dim3 grid,
							   bool prefiltered,
							   const CudaRasterizer::ColourType colourType)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	float3 curr_p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, prefiltered, curr_p_view))
		return;

	// Transform point by projecting
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	glm::mat3 canonical2view(0.f);
	compute_canonical2view(scales[idx],
						   rotations[idx],
						   p_orig,
						   glm::vec3(curr_p_view.x, curr_p_view.y, curr_p_view.z),
						   viewmatrix,
						   canonical2view,
						   normal[idx],
						   normal_sign[idx]);

	// Compute transformation matrix
	glm::mat3 T;
	if (transMat_precomp == nullptr)
	{
		compute_transmat(((float3 *)orig_points)[idx], scales[idx], rotations[idx], projmatrix, viewmatrix, W, H, T);
		float3 *T_ptr = (float3 *)transMats;
		T_ptr[idx * 3 + 0] = {T[0][0], T[0][1], T[0][2]};
		T_ptr[idx * 3 + 1] = {T[1][0], T[1][1], T[1][2]};
		T_ptr[idx * 3 + 2] = {T[2][0], T[2][1], T[2][2]};
	}
	else
	{
		glm::vec3 *T_ptr = (glm::vec3 *)transMat_precomp;
		T = glm::mat3(
			T_ptr[idx * 3 + 0],
			T_ptr[idx * 3 + 1],
			T_ptr[idx * 3 + 2]);
	}

	// Compute center and radius
	glm::vec2 point_image;
	float radius;
	{
		glm::vec2 extent;
		bool ok = compute_aabb(T, point_image, extent);
		if (!ok) return;
		radius = ceil(3.0f * max(extent.x, extent.y));
	}

	uint2 rect_min, rect_max;
	getRect(make_float2(point_image.x, point_image.y), radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	if (colors_precomp == nullptr)
	{
		glm::vec3 result = computeColorFromSH(idx,
											  D,
											  M,
											  (glm::vec3 *)orig_points,
											  *cam_pos,
											  shs,
											  clamped,
											  colourType);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	// Store some useful helper data for the next steps.
	p_view[idx] = curr_p_view;
	depths[idx] = curr_p_view.z;
	radii[idx] = (int)radius;
	points_xy_image[idx] = make_float2(point_image.x, point_image.y);
	

	// Inverse canonical2view
	// TODO Remove view2canonical altogether
	glm::mat3 R = buildRotationMatrix(rotations[idx]);
	glm::mat3 world2view = glm::mat3(
	viewmatrix[0], viewmatrix[1], viewmatrix[2],
	viewmatrix[4], viewmatrix[5], viewmatrix[6],
	viewmatrix[8], viewmatrix[9], viewmatrix[10]);
	glm::mat3x3 inverse_scale_matrix(0.f);
	inverse_scale_matrix[0][0] = 1.f / scales[idx][0];
	inverse_scale_matrix[1][1] = 1.f / scales[idx][1];
	view2canonical[idx] = inverse_scale_matrix * glm::transpose(R) * glm::transpose(world2view);

	// Inverse 2D covariance and opacity neatly pack into one float4
	const float opacity = opacities[idx];
	// TODO Remove conic opacity altogether
	conic_opacity[idx] = {normal[idx].x, normal[idx].y, normal[idx].z, opacity};
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X *BLOCK_Y)
	renderCUDA(
		const uint2 *__restrict__ ranges,
		const uint32_t *__restrict__ point_list,
		int W, int H,
		const float *__restrict__ opacity,
		const glm::vec3 *__restrict__ normal,
		const glm::vec3 *__restrict__ textureMap,
		const int2 *__restrict__ textureResolution,
		const int *__restrict__ textureMapStartingOffset,
		const float *__restrict__ texelSize,
		const float *__restrict__ viewmatrix,
		const glm::vec2 *__restrict__ scales,
		const glm::vec4 *__restrict__ rotations,
		const glm::vec3 *__restrict__ p_view,
		const glm::mat3 *__restrict__ view2canonical,
		const float2 focal,
		const glm::vec3 *__restrict__ colors,
		float *__restrict__ final_T,
		uint32_t *__restrict__ n_contrib,
		const float *__restrict__ bg_color,
		const CudaRasterizer::ColourType colourType,
		float *__restrict__ out_color,
		float *__restrict__ out_features,
		int *__restrict__ out_touched_pixels,
		float *__restrict__ out_weight,
		const bool calculate_mean_transmittance,
		const bool texture_debug_view)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = {(float)pix.x, (float)pix.y};

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ glm::vec3 collected_p_view[BLOCK_SIZE];
	__shared__ glm::mat3 collected_view2canonical[BLOCK_SIZE];
	__shared__ glm::vec2 collected_scale[BLOCK_SIZE];
	__shared__ glm::vec3 collected_colour[BLOCK_SIZE];
	__shared__ float collected_texelSize[BLOCK_SIZE];
	__shared__ float collected_opacity[BLOCK_SIZE];
	__shared__ glm::vec3 collected_normal[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	glm::vec<CHANNELS, float> C(0.f);

#ifdef REND_NORMAL
	glm::vec3 pixel_normal(0.f);
#endif
	float pixel_invdepth(0.f);

	// Ray view-space
	const glm::vec3 ray_origin_view_space(0.f);
	// Taking a point with t2=1 (third coordinate in view space) we get
	const glm::vec3 ray_direction_view = glm::normalize(glm::vec3((pixf.x - (W - 1) / 2.f) / focal.x, (pixf.y - (H - 1) / 2.f) / focal.y, 1.0f));

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_p_view[block.thread_rank()] = p_view[coll_id];
			collected_view2canonical[block.thread_rank()] = view2canonical[coll_id];
			collected_scale[block.thread_rank()] = scales[coll_id];
			collected_colour[block.thread_rank()] = colors[coll_id];
			collected_texelSize[block.thread_rank()] = texelSize[coll_id];
			collected_opacity[block.thread_rank()] = opacity[coll_id];
			collected_normal[block.thread_rank()] = normal[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			float base{0.f};

			// Load this primitive's parameters from shared memory
			const int global_id = collected_id[j];
			const glm::vec3 &curr_p_view = collected_p_view[j];
			const glm::vec2 &curr_scale = collected_scale[j];
			const float curr_opacity = collected_opacity[j];
			const glm::vec3 &curr_normal = collected_normal[j];
			
			const glm::vec3 &curr_colour = collected_colour[j];
			const int2 &curr_texture_resolution = textureResolution[global_id];
			const int &curr_texture_map_starting_offset = textureMapStartingOffset[global_id];
			const float curr_texelSize = collected_texelSize[j];

			glm::vec2 intersection_point_canonical(0.f);
			glm::vec2 intersection_point_axisaligned(0.f);
			float curr_invdepth{0.f};

			const glm::mat3 R = buildRotationMatrix(rotations[global_id]);

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
				continue;
			const glm::vec3 intersection_point_view = ray_origin_view_space + ray_depth * ray_direction_view - curr_p_view;
			curr_invdepth = 1.f / (intersection_point_view.z + curr_p_view.z);
			// TODO this could be faster I think
			intersection_point_axisaligned = glm::vec2(inv_R * inv_world2View * intersection_point_view);
			intersection_point_canonical = 1.f / curr_scale * intersection_point_axisaligned;
			base = glm::dot(intersection_point_canonical, intersection_point_canonical);

			const float power = -0.5f * base;

			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix).
			float alpha = min(0.99f, curr_opacity * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;

			float test_T = T * (1.f - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}
			const glm::vec3 &base_colour = curr_colour;
			glm::vec3 colour(base_colour);

			// TODO make that bool? No need for 2 values
			int2 clamped{0, 0};
			glm::vec2 weights(0.f);
			glm::vec<2, int> index, index_plus1, cutoff_index_start, cutoff_index_end;
			const glm::vec2 &uv = canonical2TexUV(
				curr_texture_resolution,
				curr_texelSize,
				curr_scale * (float)TEXTURE_GAUSSIAN_CUTOFF,
				intersection_point_axisaligned,
				cutoff_index_start,
				cutoff_index_end,
				clamped);

			// Interpolate
			// Colour offsets
			if (!clamped.x && !clamped.y && colourType != CudaRasterizer::ColourType::BASE)
			{
				colour += getTextureColour(
					uv,
					curr_texture_resolution,
					textureMap + curr_texture_map_starting_offset,
					cutoff_index_start,
					cutoff_index_end,
					weights,
					index,
					index_plus1,
					nullptr,
					texture_debug_view);
			}
			colour = glm::max(colour, glm::vec3(0.f));

#ifdef REND_NORMAL
				pixel_normal += curr_normal * alpha * T;
				pixel_invdepth += curr_invdepth * alpha * T;
#endif
				// // Eq. (3) from 3D Gaussian splatting paper.
				// for (int ch = 0; ch < CHANNELS; ch++)
				// {
				C += colour * alpha * T;
				// }

				if (calculate_mean_transmittance)
				{
					atomicAdd(&out_touched_pixels[global_id], 1);
					atomicAdd(&out_weight[global_id], T);
				}
				T = test_T;

				// Keep track of last range entry to update this
				// pixel.
				last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
#ifdef REND_NORMAL
		out_features[(NORMAL_OFFSET + 0) * H * W + pix_id] = pixel_normal[0];
		out_features[(NORMAL_OFFSET + 1) * H * W + pix_id] = pixel_normal[1];
		out_features[(NORMAL_OFFSET + 2) * H * W + pix_id] = pixel_normal[2];
		out_features[INVDEPTH_OFFSET * H * W + pix_id] = pixel_invdepth;
#endif
	}
			}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint32_t* point_list,
	int W, int H,
	const CudaRasterizer::TextureState& textureState,
	const CudaRasterizer::GeometryState& geomState,
	const CudaRasterizer::ImageState& imgState,
	const float2 focal,
	const float* colors,
	const float* bg_color,
	const CudaRasterizer::ColourType colourType,
	const float *viewmatrix,
	float* out_color,
	float* out_features,
	int* out_touched_pixels,
	float* out_weight,
	const bool calculate_mean_transmittance,
	const bool texture_debug_view)
{
	renderCUDA<NUM_CHANNELS><<<grid, block>>>(
		imgState.ranges,
		point_list,
		W, H,
		geomState.opacity,
		textureState.buffer.normal,
		textureState.textureMap,
		textureState.textureResolution,
		textureState.textureMapStartingOffset,
		textureState.texelSize,
		viewmatrix,
		geomState.scale,
		geomState.rotation,
		(glm::vec3 *)textureState.buffer.mean,
		textureState.buffer.view2canonical,
		focal,
		(glm::vec3 *)colors,
		imgState.accum_alpha,
		imgState.n_contrib,
		bg_color,
		colourType,
		out_color,
		out_features,
		out_touched_pixels,
		out_weight,
		calculate_mean_transmittance,
		texture_debug_view);
}

void FORWARD::preprocess(int P, const int D, int M,
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
	const CudaRasterizer::ColourType colourType)
{
	preprocessCUDA<NUM_CHANNELS><<<(P + 255) / 256, 256>>>(
		P, D, M,
		(float *)geomState.mean,
		geomState.scale,
		geomState.rotation,
		geomState.opacity,
		shs,
		transMat_precomp,
		colors_precomp,
		viewmatrix,
		projmatrix,
		cam_pos,
		W, H,
		tan_fov,
		focal,
		radii,
		geomState.buffer.clamped,
		geomState.buffer.means2D,
		geomState.buffer.depths,
		geomState.buffer.transMat,
		geomState.buffer.rgb,
		geomState.buffer.conic_opacity,
		geomState.buffer.tiles_touched,
		textureState.buffer.mean,
		textureState.buffer.view2canonical,
		textureState.buffer.normal,
		textureState.buffer.normal_sign,
		grid,
		prefiltered,
		colourType);
}
