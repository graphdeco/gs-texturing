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

#ifndef CUDA_RASTERIZER_AUXILIARY_H_INCLUDED
#define CUDA_RASTERIZER_AUXILIARY_H_INCLUDED

#include "config.h"
#include "stdio.h"
//#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>


#define PI 3.14159265358979323846f
#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#define NUM_WARPS (BLOCK_SIZE/32)
#define REND_NORMAL
#define DUAL_VISIBLE
#define SCALE_INVARIANT_EXPERIMENT
#define TEXTURE_EXTENT 2
#define NORMAL_OFFSET 0
#define INVDEPTH_OFFSET 3
#define VARIANCE_OFFSET 4
#define COLORSAMPLES_NUM_OFFSET 5
#define TEXTURE_GAUSSIAN_CUTOFF 3
// Spherical harmonics coefficients
__device__ const float SH_C0 = 0.28209479177387814f;
__device__ const float SH_C1 = 0.4886025119029199f;
__device__ const float SH_C2[] = {
	1.0925484305920792f,
	-1.0925484305920792f,
	0.31539156525252005f,
	-1.0925484305920792f,
	0.5462742152960396f
};
__device__ const float SH_C3[] = {
	-0.5900435899266435f,
	2.890611442640554f,
	-0.4570457994644658f,
	0.3731763325901154f,
	-0.4570457994644658f,
	1.445305721320277f,
	-0.5900435899266435f
};

// Normalises the given homogenous coordinate
__device__ inline glm::vec3 normaliseHomogenousPoint(const glm::vec4 &point)
{
	float norm_factor = 1.f / (point.w + 0.000001f);
	return glm::vec3(point * norm_factor);
}

// Returns the Hadamard product of a vector and a matrix
// Basically hadamard produce of each column of the matrix with the vector
__device__ inline glm::mat3 hadamardProduct(const glm::vec3 &v, const glm::mat3 &M)
{
	return glm::mat3(v * M[0],
					 v * M[1],
					 v * M[2]);
}

// Returns the Hadamard product of a matrix and a vector
// Basically hadamard produce of each row of the matrix with the vector
__device__ inline glm::mat3 hadamardProduct(const glm::mat3 &M, const glm::vec3 &v)
{
	return glm::transpose(hadamardProduct(v, glm::transpose(M)));
}

// Returns the frobenius inner product of two matrices.
// Basically an element-wise multiplication and summation of resulting elements
__device__ inline float frobeniusInnerProduct(const glm::mat3 &a, const glm::mat3 &b)
{
	float result = 0.f;

	const float* a_ptr = glm::value_ptr(a);
	const float* b_ptr = glm::value_ptr(b);

	for (int i = 0; i < 9; ++i)
	{
		result += a_ptr[i] * b_ptr[i];
	}
	return result;
}

// Returns rotation matrix from normalised quaternion
__device__ inline glm::mat3 buildRotationMatrix(const glm::vec4 q)
{
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	return glm::mat3(
		1.f - 2.f * (y * y + z * z), 	2.f * (x * y + r * z), 			2.f * (x * z - r * y),
		2.f * (x * y - r * z),			1.f - 2.f * (x * x + z * z),	2.f * (y * z + r * x),
		2.f * (x * z + r * y), 			2.f * (y * z - r * x),			1.f - 2.f * (x * x + y * y)
	);

}

// Function that does the very specific thing of calculating the outer product of two vec2
// and then returning a vec3 containing the top left element, the anti-trace (sum of the antidiagonal)
// and the bottom right element
// So, for vec1 = a b and vec2 = c d it creates the matrix
// ac    ad
// bc    bd
// and returns vec3(ac, (ad+bc), bd)
__forceinline__ __device__ glm::vec3 coefficientsFromOuterProduct(const glm::vec2 &a, const glm::vec2 &b)
{
	glm::mat2 outerProduct = glm::outerProduct(a, b);
	return glm::vec3(outerProduct[0][0], outerProduct[0][1] + outerProduct[1][0], outerProduct[1][1]);
}

__forceinline__ __device__ float ndc2Pix(float v, int S)
{
	return ((v + 1.0f) * S - 1.0f) * 0.5f;
}

__forceinline__ __device__ float calculateIntersectionRayDepth(const glm::vec3 &ray_origin, const glm::vec3 &ray_direction, const glm::vec3 &mean, const glm::vec3 &normal)
{
	return glm::dot(mean, normal) / (glm::dot(ray_direction, normal) + 1e-8f);
}

// Transform the scaled canonical intersection point to UV.
// The texture values are taken on the bottom left corner of each texel
// Since we are interpolating bilinearly, we take UV values that are up to 1 value
// away from our texture boundaries. So from -1 to maxTextureRes
__forceinline__ __device__ glm::vec2 canonical2TexUV(
	const int2 &maxTextureRes,
	const float texelSize,
	const glm::vec2 cutoff,
	const glm::vec2 canonical,
	glm::vec<2, int> &cutoff_index_start,
	glm::vec<2, int> &cutoff_index_end,
	int2 &clamped)
{
	if (abs(canonical.x) >= cutoff.x + texelSize)
	{
		clamped.x = true;
	}
	if (abs(canonical.y) >= cutoff.y + texelSize)
	{
		clamped.y = true;
	}

	const glm::vec2 indexOffset((maxTextureRes.x - 1) / 2.f, (maxTextureRes.y - 1) / 2.f);
	const glm::vec2 result = canonical / texelSize + indexOffset;
	
	// Since the cutoff might be different from 1/255 (threshold for the rendering of a gaussian
	// that is very close to 3stds), we need to calculate the cutoff indices for our texture
	cutoff_index_start = glm::max(glm::floor(-cutoff / texelSize + indexOffset), glm::vec2(0.f));
	cutoff_index_end = glm::min(glm::ceil(cutoff / texelSize + indexOffset), glm::vec2(maxTextureRes.x - 1, maxTextureRes.y - 1));

	if (result.x < -1.f || result.x >= maxTextureRes.x)
	{
		clamped.x = true;
	}
	else if (result.y < -1.f || result.y >= maxTextureRes.y)
	{
		clamped.y = true;
	}

	return glm::max(glm::min(
						result,
						glm::vec2(maxTextureRes.x - 1e-8f, maxTextureRes.y - 1e-8f)),
					glm::vec2(-1.f + 1e-8f));
}

template <uint32_t C>
// Backward for the texture interpolation function.
// It computes the derivatives for the texture features
// and for the uv coordinates
__forceinline__ __device__ void getTextureColourBackward(
	const int2 &maxtextureRes,
	const glm::vec2 cutoff_index_start,
	const glm::vec2 cutoff_index_end,
	const glm::vec2 &weights,
	const glm::vec<2, int> &index,
	const glm::vec<2, int> &index_plus1,
	const glm::vec3 *colorSamples,
	glm::mat2x3 &dcolor_duv,
	const float *dL_dcolor,
	float *dL_dtextureFeatures)
{
	dcolor_duv[0] = (1 - weights.y) * (colorSamples[1] - colorSamples[0]) + weights.y * (colorSamples[3] - colorSamples[2]);
	dcolor_duv[1] = (1 - weights.x) * (colorSamples[2] - colorSamples[0]) + weights.x * (colorSamples[3] - colorSamples[1]);

	// Iterate over the 4 samples surrounding our sample point
	for (int i = 0; i <= 1; ++i)
	{
		const int indexX = index.x * (1 - i) + index_plus1.x * i;
		for (int j = 0; j <= 1; ++j)
		{
			const int indexY = index.y * (1 - j) + index_plus1.y * j;

			if (!(indexX >= cutoff_index_start.x && indexX <= cutoff_index_end.x &&
				  indexY >= cutoff_index_start.y && indexY <= cutoff_index_end.y))
				continue;

			// This term is common on all colour channels
			const float dcolor_dtextureFeatures = ((1 - weights.x) * (1 - i) + weights.x * i) *
												  ((1 - weights.y) * (1 - j) + weights.y * j);
			const int offset = indexX * maxtextureRes.y + indexY;
			for (int ch = 0; ch < C; ++ch)
			{
				atomicAdd(dL_dtextureFeatures + offset * C + ch, dL_dcolor[ch] * dcolor_dtextureFeatures);
			}
		}
	}
}

__forceinline__ __device__ glm::vec3 getTextureColour(
	const glm::vec2 uv,
	const int2 &maxTextureRes,
	const glm::vec3 *textureMap,
	const glm::vec2 cutoff_index_start,
	const glm::vec2 cutoff_index_end,
	glm::vec2 &weights,
	glm::vec<2, int> &index,
	glm::vec<2, int> &index_plus1,
	glm::vec3 *colorsSamples = nullptr,
	const bool texture_debug_view = false)
{
	// A bit convoluted, but x here is the height, and y is the width of the texture map

	const glm::vec2 &fIndex = uv;
	index = glm::floor(fIndex);
	weights = fIndex - glm::vec2(index);
	index_plus1 = index + 1;

	if(texture_debug_view)
	{
		// Closest neighbour
		int indexX = min(max((int)round(fIndex.x), 0), maxTextureRes.x - 1);
		int indexY = min(max((int)round(fIndex.y), 0), maxTextureRes.y - 1);
		return textureMap[indexY + maxTextureRes.y * indexX];
	}

	// Bi-linear Interpolation
	glm::vec3 c00(0.f);
	glm::vec3 c10(0.f);
	glm::vec3 c01(0.f);
	glm::vec3 c11(0.f);

	if (index.x >= cutoff_index_start.x && index.y >= cutoff_index_start.y)
		c00 = textureMap[index.y + maxTextureRes.y * index.x];
	if (index_plus1.x <= cutoff_index_end.x && index.y >= cutoff_index_start.y)
		c10 = textureMap[index.y + maxTextureRes.y * index_plus1.x];
	if (index.x >= cutoff_index_start.x && index_plus1.y <= cutoff_index_end.y)
		c01 = textureMap[index_plus1.y + maxTextureRes.y * index.x];
	if (index_plus1.x <= cutoff_index_end.x && index_plus1.y <= cutoff_index_end.y)
		c11 = textureMap[index_plus1.y + maxTextureRes.y * index_plus1.x];

	glm::vec3 c0 = c00 * (1.f - weights.x) + c10 * weights.x;
	glm::vec3 c1 = c01 * (1.f - weights.x) + c11 * weights.x;

	glm::vec3 c = c0 * (1.f - weights.y) + c1 * weights.y;

	if(colorsSamples != nullptr)
	{
		colorsSamples[0] = c00;
		colorsSamples[1] = c10;
		colorsSamples[2] = c01;
		colorsSamples[3] = c11;
	}

	return c;
}

__forceinline__ __device__ void getRect(const float2 p, int max_radius, uint2& rect_min, uint2& rect_max, dim3 grid)
{
	rect_min = {
		min(grid.x, max((int)0, (int)((p.x - max_radius) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y - max_radius) / BLOCK_Y)))
	};
	rect_max = {
		min(grid.x, max((int)0, (int)((p.x + max_radius + BLOCK_X - 1) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y + max_radius + BLOCK_Y - 1) / BLOCK_Y)))
	};
}

__forceinline__ __device__ float3 transformPoint3x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[3] * p.y + matrix[6] * p.z,
		matrix[1] * p.x + matrix[4] * p.y + matrix[7] * p.z,
		matrix[2] * p.x + matrix[5] * p.y + matrix[8] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float3 transformPoint4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
	};
	return transformed;
}

__forceinline__ __device__ float4 transformPoint4x4(const float3& p, const float* matrix)
{
	float4 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
		matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z,
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z,
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3Transpose(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z,
		matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z,
		matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float dnormvdz(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);
	float dnormvdz = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdz;
}

__forceinline__ __device__ float3 dnormvdv(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float3 dnormvdv;
	dnormvdv.x = ((+sum2 - v.x * v.x) * dv.x - v.y * v.x * dv.y - v.z * v.x * dv.z) * invsum32;
	dnormvdv.y = (-v.x * v.y * dv.x + (sum2 - v.y * v.y) * dv.y - v.z * v.y * dv.z) * invsum32;
	dnormvdv.z = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float4 dnormvdv(float4 v, float4 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float4 vdv = { v.x * dv.x, v.y * dv.y, v.z * dv.z, v.w * dv.w };
	float vdv_sum = vdv.x + vdv.y + vdv.z + vdv.w;
	float4 dnormvdv;
	dnormvdv.x = ((sum2 - v.x * v.x) * dv.x - v.x * (vdv_sum - vdv.x)) * invsum32;
	dnormvdv.y = ((sum2 - v.y * v.y) * dv.y - v.y * (vdv_sum - vdv.y)) * invsum32;
	dnormvdv.z = ((sum2 - v.z * v.z) * dv.z - v.z * (vdv_sum - vdv.z)) * invsum32;
	dnormvdv.w = ((sum2 - v.w * v.w) * dv.w - v.w * (vdv_sum - vdv.w)) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float sigmoid(float x)
{
	return 1.0f / (1.0f + expf(-x));
}

__forceinline__ __device__ bool in_frustum(int idx,
	const float* orig_points,
	const float* viewmatrix,
	bool prefiltered,
	float3& p_view)
{
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	p_view = transformPoint4x3(p_orig, viewmatrix);

	if (p_view.z <= 0.2f)
	{
		if (prefiltered)
		{
			printf("Point is filtered although prefiltered is set. This shouldn't happen!");
			__trap();
		}
		return false;
	}
	return true;
}

// Returns rotation matrix from normalised quaternion
__device__ inline glm::mat3 buildJacobianMatrix(const float3 &mean,
												const float* viewmatrix,
												const float2 &tan_fov,
												const float2 &focal)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002).
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fov.x;
	const float limy = 1.3f * tan_fov.y;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	// const float inv_length = 1 / sqrt(t.x * t.x + t.y * t.y + t.z * t.z);

	// Jacobian in pixel space (focal in pixels)
	return glm::mat3(
		focal.x / t.z, 0.0f, 0.,
		0.0f, focal.y / t.z, 0.,
		-(focal.x * t.x) / (t.z * t.z), -(focal.y * t.y) / (t.z * t.z), 0.01f / (t.z * t.z));
	// TODO FIX 0.01 WITH THE ACTUAL VALUE TAKEN FROM CAMERA INTRINSICS
}

#define CHECK_CUDA(A, debug) \
A; if(debug) { \
auto ret = cudaDeviceSynchronize(); \
if (ret != cudaSuccess) { \
std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
throw std::runtime_error(cudaGetErrorString(ret)); \
} \
}

template <class element_t, class vector_t>
using enable_if_element_vector_types_compatible_t = std::enable_if_t<sizeof(vector_t) % sizeof(element_t) == 0>;

template <
    typename T,
    typename element_t = float,
    typename           = enable_if_element_vector_types_compatible_t<element_t, T>>
__device__ T atomicAddVector(T *adress, const T &val) {
    constexpr auto n_elements       = sizeof(T) / sizeof(element_t);
    auto           address_elements = reinterpret_cast<element_t *>(adress);
    auto           val_elements     = reinterpret_cast<const element_t *>(&val);

#pragma unroll
    for (size_t i = 0; i < n_elements; i++)
        atomicAdd(address_elements + i, val_elements[i]);
}

inline __device__ glm::mat3
scale_to_mat(const glm::vec2 scale) {
	glm::mat3 S = glm::mat3(1.f);
	S[0][0] = scale.x;
	S[1][1] = scale.y;
	return S;
}


#endif