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

#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Backward pass that calculates partial derivative of the intersection point in UV coordinates
// with respect to the intersection point in canonical space
__device__ void canonical2TexUVBackward(
	const float texelSize,
	const int2 &clamped_uv,
	glm::mat2 &duv_dintersection_point_axisaligned)
{
	if (!clamped_uv.x)
		duv_dintersection_point_axisaligned[0][0] = 1.f / texelSize;
	if (!clamped_uv.y)
		duv_dintersection_point_axisaligned[1][1] = 1.f / texelSize;
}

// Calculates the derivatives of the intersection point in view space
// w.r.t. the mean and normal vectors in view space
__device__ void intersectionPointViewBackward(
	const glm::vec3 &dL_dview,
	const glm::vec3 &ray_direction,
	const glm::vec3 &normal,
	const glm::vec3 &mean,
	glm::vec3 &dL_dmeanview,
	glm::vec3 &dL_dnormalview)
{
	// xv = ray * ray_depth + r_origin - meanv
	const glm::vec3 &dview_dray_depth = ray_direction;
	const float inv_dot_ray_normal = 1.f / glm::dot(ray_direction, normal);
	const float dot_mean_normal = glm::dot(mean, normal);

	dL_dmeanview += dL_dview * (inv_dot_ray_normal * glm::outerProduct(dview_dray_depth, normal) - glm::mat3(1.f));

	const glm::vec3 dray_depth_dnormal = (mean - inv_dot_ray_normal * dot_mean_normal * ray_direction);

	dL_dnormalview += dL_dview * (inv_dot_ray_normal * glm::outerProduct(dview_dray_depth, dray_depth_dnormal));
}
__device__ void view2CanonicalScaleBackward(
	const glm::vec2 &scales,
	const glm::vec2 &intersectionPointAxisAligned,
	glm::mat2 &dintersection_point_canonical_dscale)
{
	dintersection_point_canonical_dscale[0][0] = -1.f/(scales[0]*scales[0]) * intersectionPointAxisAligned[0];
	dintersection_point_canonical_dscale[1][1] = -1.f/(scales[1]*scales[1]) * intersectionPointAxisAligned[1];
}

__device__ void view2CanonicalRotationBackward(
	const glm::vec3 &dL_dintersection_point_axisaligned,
	const glm::vec3 &intersection_point_world,
	glm::mat3 &dL_dR)
{
	dL_dR = glm::outerProduct(intersection_point_world, dL_dintersection_point_axisaligned);
}


// Backward pass that calculates partial derivative of the gaussian falloff to 
// the scales
__device__ void falloffCanonicalBackward(
	const float gaussian_falloff,
	const glm::vec2 &intersection_point_canonical,
	glm::vec2 &dG_dcanonical
)
{
	dG_dcanonical = -gaussian_falloff * intersection_point_canonical;
}

// Backward pass that calculates partial derivative of the rotation matrix
// to the quaternions
__device__ void rotationQuaternionBackward(
	const glm::mat3 &dL_dR,
	const glm::vec4 &q,
	glm::vec4 &dL_dquaternion
)
{
	// Quaternion is normalised
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	glm::mat3 dR_dqr{
		0.f, 2.f * z, -2.f * y,
		-2.f * z, 0.f, 2.f * x,
		2.f * y, -2.f * x, 0.f};
	glm::mat3 dR_dqx{
		0.f, 2.f * y, 2.f * z,
		2.f * y, -4.f * x, 2.f * r,
		2.f * z, -2.f * r, -4.f * x};
	glm::mat3 dR_dqy{
		-4.f * y, 2.f * x, -2.f * r,
		2.f * x, 0.f, 2.f * z,
		2.f * r, 2.f * z, -4.f * y};
	glm::mat3 dR_dqz{
		-4.f * z, 2.f * r, 2.f * x,
		-2.f * r, -4.f * z, 2.f * y,
		2.f * x, 2.f * y, 0.f};

	dL_dquaternion[0] = frobeniusInnerProduct(dL_dR, dR_dqr);
	dL_dquaternion[1] = frobeniusInnerProduct(dL_dR, dR_dqx);
	dL_dquaternion[2] = frobeniusInnerProduct(dL_dR, dR_dqy);
	dL_dquaternion[3] = frobeniusInnerProduct(dL_dR, dR_dqz);
}

// Backward pass for conversion of spherical harmonics to RGB for
// each Gaussian.
__device__ void computeColorFromSH(int idx,
								   const int deg,
								   int max_coeffs,
								   const glm::vec3 *means,
								   glm::vec3 campos,
								   const float *shs,
								   const bool *clamped,
								   const glm::vec3 *dL_dcolor,
								   glm::vec3 *dL_dmeans,
								   glm::vec3 *dL_dshs)
{
	// Compute intermediate values, as it is done during forward
	glm::vec3 pos = means[idx];
	glm::vec3 dir_orig = pos - campos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	glm::vec3 dL_dRGB = dL_dcolor[idx];
	dL_dRGB.x *= clamped[3 * idx + 0] ? 0.f : 1.f;
	dL_dRGB.y *= clamped[3 * idx + 1] ? 0.f : 1.f;
	dL_dRGB.z *= clamped[3 * idx + 2] ? 0.f : 1.f;

	glm::vec3 dRGBdx(0.f);
	glm::vec3 dRGBdy(0.f);
	glm::vec3 dRGBdz(0.f);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// Target location for this Gaussian to write SH gradients to
	glm::vec3* dL_dsh = dL_dshs + idx * max_coeffs;

	// No tricks here, just high school-level calculus.
	float dRGBdsh0 = SH_C0;
	dL_dsh[0] = dRGBdsh0 * dL_dRGB;
	if (deg > 0)
	{
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;

		dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		dL_dsh[3] = dRGBdsh3 * dL_dRGB;
		
		dRGBdx = -SH_C1 * sh[3];
		dRGBdy = -SH_C1 * sh[1];
		dRGBdz = SH_C1 * sh[2];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);

			dL_dsh[4] = dRGBdsh4 * dL_dRGB;
			dL_dsh[5] = dRGBdsh5 * dL_dRGB;
			dL_dsh[6] = dRGBdsh6 * dL_dRGB;
			dL_dsh[7] = dRGBdsh7 * dL_dRGB;
			dL_dsh[8] = dRGBdsh8 * dL_dRGB;

			dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
			dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
			dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				
				dL_dsh[9] = dRGBdsh9 * dL_dRGB;
				dL_dsh[10] = dRGBdsh10 * dL_dRGB;
				dL_dsh[11] = dRGBdsh11 * dL_dRGB;
				dL_dsh[12] = dRGBdsh12 * dL_dRGB;
				dL_dsh[13] = dRGBdsh13 * dL_dRGB;
				dL_dsh[14] = dRGBdsh14 * dL_dRGB;
				dL_dsh[15] = dRGBdsh15 * dL_dRGB;

				dRGBdx += (
					SH_C3[0] * sh[9] * 3.f * 2.f * xy +
					SH_C3[1] * sh[10] * yz +
					SH_C3[2] * sh[11] * -2.f * xy +
					SH_C3[3] * sh[12] * -3.f * 2.f * xz +
					SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
					SH_C3[5] * sh[14] * 2.f * xz +
					SH_C3[6] * sh[15] * 3.f * (xx - yy));

				dRGBdy += (
					SH_C3[0] * sh[9] * 3.f * (xx - yy) +
					SH_C3[1] * sh[10] * xz +
					SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
					SH_C3[3] * sh[12] * -3.f * 2.f * yz +
					SH_C3[4] * sh[13] * -2.f * xy +
					SH_C3[5] * sh[14] * -2.f * yz +
					SH_C3[6] * sh[15] * -3.f * 2.f * xy);

				dRGBdz += (
					SH_C3[1] * sh[10] * xy +
					SH_C3[2] * sh[11] * 4.f * 2.f * yz +
					SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
					SH_C3[4] * sh[13] * 4.f * 2.f * xz +
					SH_C3[5] * sh[14] * (xx - yy));
			}
		}
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
	glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));

	// Account for normalization of direction
	float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the view-dependent color.
	// Additional mean gradient is accumulated in below methods.
	dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
}

// Computes the derivative of the inverse 2D covariance matrix
// w.r.t. to the 2d covariance matrix
// Essentially, applies dinvA = -inv(A) * dA * inv(A)
__device__ void inverseCov2DBackward(
	const float denomSqInv,
	const glm::vec3 &A_entries,
	const glm::vec3 &invA_entries,
	glm::vec3 &dL_dA
)
{
	// Gradients of loss w.r.t. entries of 2D covariance matrix,
	// given gradients of loss w.r.t. conic matrix (inverse covariance matrix).
	// e.g., dL / da = dL / d_conic_a * d_conic_a / d_a

	const float a = A_entries.x;
	const float b = A_entries.y;
	const float c = A_entries.z;

	const float x = invA_entries.x;
	const float y = invA_entries.y;
	const float z = invA_entries.z;

	dL_dA.x = denomSqInv * (-c * c * x + 2.f * b * c * y - b * b * z);
	dL_dA.y = denomSqInv * (+b * c * x - (a * c + b * b) * y + a * b * z);
	dL_dA.z = denomSqInv * (-b * b * x + 2.f * a * b * y - a * a * z);
}

// Computes the derivative of the 2D covariance matrix
// in pixel space (that is, after view and jacobian of the projection have been applied
// and the 3rd row and column have been dropped
// I(2x3) * J * W * Cov3D * W^T * J * I(3x2))
// w.r.t. Cov3D, which is the covariance in world space
__device__ void cov3D2Cov2DPixelSpaceBackward(
	const glm::mat3 &T,
	const glm::vec3 &dL_dcov2D_vec,
	float *dL_dcov)
{
		// dL_dcov is a symmetric matrix equal to
		// T(first two rows)^T dL_dcov2D_vec T(first two rows)
		// We apply some linear algebra magic to avoid some computations,
		// bectionrectly the coPixelSpaceefficients of the 3 different elements of dL_cov2D()
		// that are needed for each one of the 6 different elements of dL_cov
		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
		// given gradients w.r.t. 2D covariance matrix (diagonal).
		// cov2D = T * Vrk * transpose(T);
		dL_dcov[0] = glm::dot(coefficientsFromOuterProduct(T[0], T[0]), dL_dcov2D_vec);
		dL_dcov[3] = glm::dot(coefficientsFromOuterProduct(T[1], T[1]), dL_dcov2D_vec);
		dL_dcov[5] = glm::dot(coefficientsFromOuterProduct(T[2], T[2]), dL_dcov2D_vec);

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry,
		// given gradients w.r.t. 2D covariance matrix (off-diagonal).
		// cov2D = T * Vrk * transpose(T);
		dL_dcov[1] = glm::dot(coefficientsFromOuterProduct(T[0], T[1]), dL_dcov2D_vec);
		dL_dcov[2] = glm::dot(coefficientsFromOuterProduct(T[0], T[2]), dL_dcov2D_vec);
		dL_dcov[4] = glm::dot(coefficientsFromOuterProduct(T[1], T[2]), dL_dcov2D_vec);
}

// Computes the derivative of the cov2D in pixel space
// w.r.t. the projection matrix (T = J*W)
__device__ void projection2Cov2DPixelSpaceBackward(
	const glm::mat2 &dL_dcov2D,
	const glm::mat3x2 &T,
	const glm::mat3 &cov3D,
	glm::mat3x2 &dL_dT)
{
	// Gradients of loss w.r.t. upper 2x3 portion of intermediate matrix T
	// cov2D = T * Vrk * transpose(T);
	// dL_dT = 2 * dL_dcov2D(top left in a 3x3 matrix with rest being zeros) * T * Vrk
	// dL_dT is a 3x3 matrix but as the last row is zero,
	// we skip it to avoid some computation
	dL_dT = 2.f * dL_dcov2D * T * cov3D;
}

// Gradients of loss w.r.t. Jacobian matrix
// T = J * W
// dL_dJ = dL_dT * W^T
__device__ void jacobianBackward(
	const glm::mat3x2 &dL_dT,
	const glm::mat3 &view_matrix,
	const glm::vec3 &mean_view,
	const float2 &grad_mul,
	const float2 &focal,
	glm::vec3 &dL_dmean)
{
	glm::mat3x2 dL_dJ = dL_dT * glm::transpose(view_matrix);

	float inv_tz = 1.f / mean_view.z;
	float inv_tzsq = inv_tz * inv_tz;
	float inv_tzcb = inv_tzsq * inv_tz;

	// Gradients of loss w.r.t. transformed Gaussian mean t
	// The dJ_dtx and dJ_dty matrices are very sparse (only one non zero element)
	// So we directly compute the Frobenius inner product
	float dL_dtx = dL_dJ[2][0] * grad_mul.x * (-focal.x * inv_tzsq);
	float dL_dty = dL_dJ[2][1] * grad_mul.y * (-focal.y * inv_tzsq);
	float dL_dtz = dL_dJ[0][0] * (-focal.x * inv_tzsq) + dL_dJ[1][1] * (-focal.y * inv_tzsq)
	+ dL_dJ[2][0] * 2.f * focal.x * mean_view.x * inv_tzcb
	+ dL_dJ[2][1] * 2.f * focal.y * mean_view.y * inv_tzcb;

	// Account for transformation of mean to t
	// t = Wm
	// dL_dm = dL_dt W
	dL_dmean = glm::vec3(dL_dtx, dL_dty, dL_dtz) * view_matrix;
}

// Backward version of INVERSE 2D covariance matrix computation
// (due to length launched as separate kernel before other 
// backward steps contained in preprocess)
__global__ void computeCov2DCUDA(int P,
	const float3* means,
	const int* radii,
	const float* cov3Ds,
	const float2 focal,
	const float2 tan_fov,
	const float* viewmatrix,
	const float3* dL_dconic2D,
	glm::vec3* dL_dmeans,
	float* dL_dcov)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	// Reading location of 3D covariance for this Gaussian
	const float* cov3D = cov3Ds + 6 * idx;

	// Fetch gradients, recompute 2D covariance and relevant 
	// intermediate forward results needed in the backward.
	const float3 &curr_mean = means[idx];
	const float3 &curr_dL_dconic2D = dL_dconic2D[idx];
	float3 t = transformPoint4x3(curr_mean, viewmatrix);
	
	const float limx = 1.3f * tan_fov.x;
	const float limy = 1.3f * tan_fov.y;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;
	
	const float x_grad_mul = txtz < -limx || txtz > limx ? 0.f : 1.f;
	const float y_grad_mul = tytz < -limy || tytz > limy ? 0.f : 1.f;

	glm::mat3 J = glm::mat3(
		focal.x / t.z, 						0.0f, 								0.f,
		0.0f, 								focal.y / t.z, 						0.f,
		-(focal.x * t.x) / (t.z * t.z), 	-(focal.y * t.y) / (t.z * t.z), 	0.01f / (t.z * t.z));
		//TODO FIX 0.01 WITH THE ACTUAL VALUE TAKEN FROM CAMERA INTRINSICS

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[1], viewmatrix[2],
		viewmatrix[4], viewmatrix[5], viewmatrix[6],
		viewmatrix[8], viewmatrix[9], viewmatrix[10]);

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 T = J * W;

	glm::mat3 cov2D = T * Vrk * glm::transpose(T);

	// Use helper variables for 2D covariance entries. More compact.
	float a = cov2D[0][0] += 0.3f;
	float b = cov2D[0][1];
	float c = cov2D[1][1] += 0.3f;

	float denom = a * c - b * b;
	glm::vec3 dL_dcov2D_vec{0.f};
	float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);

	// For some of the following gradients we need only the first two rows of the T matrix,
	// as the cov2D uses only the top left part of the cov3D matrix
	glm::mat3x2 T_truncated = glm::mat3x2(
		glm::vec2(T[0][0], T[0][1]),
		glm::vec2(T[1][0], T[1][1]),
		glm::vec2(T[2][0], T[2][1])
	);

	if (denom2inv != 0)
	{
		inverseCov2DBackward(denom2inv, glm::vec3(a, b, c), glm::vec3(curr_dL_dconic2D.x, curr_dL_dconic2D.y, curr_dL_dconic2D.z), dL_dcov2D_vec);
		cov3D2Cov2DPixelSpaceBackward(T, dL_dcov2D_vec, dL_dcov + 6 * idx);
	}
	else
	{
		for (int i = 0; i < 6; i++)
			dL_dcov[6 * idx + i] = 0.f;
	}

	glm::mat3x2 dL_dT(0.f);
	projection2Cov2DPixelSpaceBackward(glm::mat2(dL_dcov2D_vec.x, dL_dcov2D_vec.y, dL_dcov2D_vec.y, dL_dcov2D_vec.z), T_truncated, Vrk, dL_dT);
	

	glm::vec3 dL_dmean(0.f);
	jacobianBackward(dL_dT,
					 W,
					 glm::vec3(t.x, t.y, t.z),
					 make_float2(x_grad_mul, y_grad_mul),
					 focal,
					 dL_dmean);

	// Gradients of loss w.r.t. Gaussian means, but only the portion
	// that is caused because the mean affects the covariance matrix.
	// Additional mean gradient is accumulated in BACKWARD::preprocess.
	dL_dmeans[idx] = dL_dmean;
}

// Backward pass for the conversion of scale and rotation to a 
// 3D covariance matrix for each Gaussian. 
__device__ void computeCov3D(int idx, const glm::vec3 scale, const glm::vec4 rot, const float* dL_dcov3Ds, glm::vec3* dL_dscales, glm::vec4* dL_drots)
{
	// Recompute (intermediate) results for the 3D covariance computation.
	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot; // / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;
	// Compute rotation matrix from quaternion
	glm::mat3 R = buildRotationMatrix(q);

	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = scale.x;
	S[1][1] = scale.y;
	S[2][2] = scale.z;

	glm::mat3 M = R * S;

	const float* curr_dL_dcov3D = dL_dcov3Ds + 6 * idx;
	// Convert per-element covariance loss gradients to matrix form
	const glm::mat3 dL_dcov3D{
		curr_dL_dcov3D[0], curr_dL_dcov3D[1], curr_dL_dcov3D[2],
		curr_dL_dcov3D[1], curr_dL_dcov3D[3], curr_dL_dcov3D[4],
		curr_dL_dcov3D[2], curr_dL_dcov3D[4], curr_dL_dcov3D[5]
	};

	// Compute loss gradient w.r.t. matrix M
	// cov3D = M * M^T
	glm::mat3 dL_dM = 2.0f * dL_dcov3D * M;

	// Gradients of loss w.r.t. scale
	glm::vec3 dL_dscale(0.f);
	dL_dscale.x = glm::dot(R[0], dL_dM[0]);
	dL_dscale.y = glm::dot(R[1], dL_dM[1]);
	dL_dscale.z = glm::dot(R[2], dL_dM[2]);

	dL_dscales[idx] += dL_dscale;

	// Compute loss gradient w.r.t. matrix R
	// dL_dR = dL_dM * S
	// Make use of the fact that S matrix is diagonal
	glm::mat3 dL_dR = glm::mat3(
		dL_dM[0] * scale.x,
		dL_dM[1] * scale.y,
		dL_dM[2] * scale.y);

	glm::mat3 dR_dqr{
		0.f, 2.f * z, -2.f * y,
		-2.f * z, 0.f, 2.f * x,
		2.f * y, -2.f * x, 0.f};
	glm::mat3 dR_dqx{
		0.f, 2.f * y, 2.f * z,
		2.f * y, -4.f * x, 2.f * r,
		2.f * z, -2.f * r, -4.f * x};
	glm::mat3 dR_dqy{
		-4.f * y, 2.f * x, -2.f * r,
		2.f * x, 0.f, 2.f * z,
		2.f * r, 2.f * z, -4.f * y};
	glm::mat3 dR_dqz{
		-4.f * z, 2.f * r, 2.f * x,
		-2.f * r, -4.f * z, 2.f * y,
		2.f * x, 2.f * y, 0.f};

	// Gradients of loss w.r.t. normalized quaternion
	glm::vec4 dL_dq;
	dL_dq.x = frobeniusInnerProduct(dL_dR, dR_dqr);
	dL_dq.y = frobeniusInnerProduct(dL_dR, dR_dqx);
	dL_dq.z = frobeniusInnerProduct(dL_dR, dR_dqy);
	dL_dq.w = frobeniusInnerProduct(dL_dR, dR_dqz);

	dL_drots[idx] += dL_dq;
}

// Backward pass of the preprocessing steps, except
// for the covariance computation and inversion
// (those are handled by a previous kernel call)
template<int C>
__global__ void preprocessCUDA(
	int P, const int D, int M,
	const float3* means,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const float* proj,
	const glm::vec3* campos,
	glm::vec3* dL_dmeans,
	float* dL_dcolor,
	float* dL_dsh)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	float3 m = means[idx];

	// Taking care of gradients from the screenspace points
	float4 m_hom = transformPoint4x4(m, proj);
	float inv_w = 1.0f / (m_hom.w + 0.0000001f);
	float inv_wsq = inv_w * inv_w;

	// Compute loss gradient w.r.t. 3D means due to gradients of 2D means
	// from rendering procedure
	// dL_dmean = dL_dmean2D * I(2x3) * dmean3Dpixelspace_dmeanhom * dmeanhom_dmean3D
	// With some linear algebra magic and leveraging that 1 row is basically skipped
	// by the I(2x3) and the sparseness of dmean3Dpixelspace_dmeanhom
	// we compute the derivatives directly

	// glm::vec3 dL_dmean;
	// dL_dmean.x = dL_dmean2D[idx].x * (proj[0] * inv_w - proj[3] * m_hom.x * inv_wsq) +
	// 			 dL_dmean2D[idx].y * (proj[1] * inv_w - proj[3] * m_hom.y * inv_wsq);
	// dL_dmean.y = dL_dmean2D[idx].x * (proj[4] * inv_w - proj[7] * m_hom.x * inv_wsq) +
	// 			 dL_dmean2D[idx].y * (proj[5] * inv_w - proj[7] * m_hom.y * inv_wsq);
	// dL_dmean.z = dL_dmean2D[idx].x * (proj[8] * inv_w - proj[11] * m_hom.x * inv_wsq) +
	// 			 dL_dmean2D[idx].y * (proj[9] * inv_w - proj[11] * m_hom.y * inv_wsq);

	// // That's the second part of the mean gradient. Previous computation
	// // of cov2D and following SH conversion also affects it.
	// dL_dmeans[idx] += dL_dmean;

	// Compute gradient updates due to computing colors from SHs
	if (shs)
		computeColorFromSH(idx, D, M, (glm::vec3*)means, *campos, shs, clamped, (glm::vec3*)dL_dcolor, (glm::vec3*)dL_dmeans, (glm::vec3*)dL_dsh);
}

// Backward version of the rendering procedure.
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X *BLOCK_Y)
	renderCUDA(
		const int n_primitives,
		const uint2 *__restrict__ ranges,
		const uint32_t *__restrict__ point_list,
		int W, int H,
		const float *__restrict__ bg_color,
		const float2 *__restrict__ points_xy_image,
		const float *__restrict__ opacity,
		const glm::vec3 *__restrict__ normal,
		const glm::vec3 *__restrict__ colors,
		const float *__restrict__ final_Ts,
		const uint32_t *__restrict__ n_contrib,
		const glm::vec3 *__restrict__ textureMap,
		const int2 *__restrict__ textureResolution,
		const int *__restrict__ textureMapStartingOffset,
		const float *__restrict__ texelSize,
		const glm::vec3 *__restrict__ p_view,
		const glm::mat3 *__restrict__ view2canonical,
		const int *__restrict__ normal_sign,
		const float2 focal,
		const float2 tan_fov,
		const float *__restrict__ viewmatrix,
		const glm::vec2 *__restrict__ scales,
		const glm::vec4 *__restrict__ rotations,
		glm::vec2 *__restrict__ dL_dscale,
		glm::vec4 *__restrict__ dL_dquaternion,
		glm::vec3 *__restrict__ dL_dmean3D,
		const float *__restrict__ dL_dpixels,
		const float *__restrict__ dL_dout_features,
		float *__restrict__ dL_dopacity,
		float *__restrict__ dL_dcolors,
		float *__restrict__ dL_dtextureFeatures)
{
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x, (float)pix.y };

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
	__shared__ int collected_normal_sign[BLOCK_SIZE];

	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors. 
	const float T_final = inside ? final_Ts[pix_id] : 0.f;
	float T = T_final;

	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	float accum_rec[C] = { 0.f };
	float dL_dpixel[C] = { 0.f };

	float dLdepth_dsplatted_invdepth(0.f);
	float accum_depth_rec(0.f);
	float last_invdepth(0.f);
#ifdef REND_NORMAL
	glm::vec3 dLnormal_dsplatted_normal(0.f);
	glm::vec3 last_normal(0.f);
	glm::vec3 accum_normal_rec(0.f);
#endif
	if (inside)
	{
		dLdepth_dsplatted_invdepth = dL_dout_features[INVDEPTH_OFFSET * H * W + pix_id];
		dLnormal_dsplatted_normal = glm::vec3(dL_dout_features[(NORMAL_OFFSET + 0) * H * W + pix_id], dL_dout_features[(NORMAL_OFFSET + 1) * H * W + pix_id], dL_dout_features[(NORMAL_OFFSET + 2) * H * W + pix_id]);
		for (int i = 0; i < C; i++)
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];
	}

	float last_alpha = 0.f;
	float last_color[C] = { 0.f };

	// Gradient of pixel coordinate w.r.t. normalized 
	// screen-space viewport corrdinates (-1 to 1)
	const float ddelx_dx = 0.5f * W;
	const float ddely_dy = 0.5f * H;

	// Ray view-space
	const glm::vec3 ray_origin_view_space(0.f);
	// Taking a point with t2=1 (third coordinate in view space) we get
	const glm::vec3 ray_direction_view = glm::normalize(glm::vec3((pixf.x - (W - 1) / 2.f) / focal.x, (pixf.y - (H - 1) / 2.f) / focal.y, 1.0f));

	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in reverse order.
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[block.thread_rank()] = coll_id;
			collected_p_view[block.thread_rank()] = p_view[coll_id];
			collected_view2canonical[block.thread_rank()] = view2canonical[coll_id];
			collected_scale[block.thread_rank()] = scales[coll_id];
			collected_colour[block.thread_rank()] = colors[coll_id];
			collected_texelSize[block.thread_rank()] = texelSize[coll_id];
			collected_opacity[block.thread_rank()] = opacity[coll_id];
			collected_normal[block.thread_rank()] = normal[coll_id];
			collected_normal_sign[block.thread_rank()] = normal_sign[coll_id];
		}
		block.sync();

		// Iterate over Gaussians
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			float curr_invdepth(0.f);
			// Keep track of current Gaussian ID. Skip, if this one
			// is behind the last contributor for this pixel.
			contributor--;
			if (contributor >= last_contributor)
				continue;

			float base{0.f};

			const int global_id = collected_id[j];
			const glm::vec3 &curr_p_view = collected_p_view[j];
			const glm::vec2 &curr_scale = collected_scale[j];
			const float curr_opacity = collected_opacity[j];
			const glm::vec3 &curr_normal = collected_normal[j];
			const int curr_normal_sign = collected_normal_sign[j];

			const glm::vec3 &curr_colour = collected_colour[j];
			const int2 &curr_texture_resolution = textureResolution[global_id];
			const int &curr_texture_map_starting_offset = textureMapStartingOffset[global_id];
			const float curr_texelSize = collected_texelSize[j];

			glm::vec3 intersection_point_view(0.f);
			glm::vec2 intersection_point_axisaligned(0.f);
			glm::vec2 intersection_point_canonical(0.f);

			const glm::mat3 R = buildRotationMatrix(rotations[global_id]);

			// R is orthonormal, so R^-1 = R^T
			const glm::mat3 inv_R = glm::transpose(R);

			const glm::mat3 &world2view = glm::mat3(
				viewmatrix[0], viewmatrix[1], viewmatrix[2],
				viewmatrix[4], viewmatrix[5], viewmatrix[6],
				viewmatrix[8], viewmatrix[9], viewmatrix[10]);

			// world2View is orthonormal, so world2View^-1 = world2View^T
			const glm::mat3 &inv_world2View = glm::transpose(world2view);

			// x = ray_origin + t ray
			// <(x - p_view), normal> = 0
			const float ray_depth = calculateIntersectionRayDepth(
				ray_origin_view_space,
				ray_direction_view,
				curr_p_view,
				curr_normal);
			// TODO ADD NEAR_PLANE HERE
			if (ray_depth < 0.2f)
				continue;
			intersection_point_view = ray_origin_view_space + ray_depth * ray_direction_view - curr_p_view;
			curr_invdepth = 1.f / (intersection_point_view.z + curr_p_view.z);
			intersection_point_axisaligned = glm::vec2(inv_R * inv_world2View * intersection_point_view);
			intersection_point_canonical = 1.f / curr_scale * intersection_point_axisaligned;

			base = glm::dot(intersection_point_canonical, intersection_point_canonical);
			const float power = -0.5f * base;

			if (power > 0.0f)
				continue;

			const float G = exp(power);
			const float alpha = min(0.99f, curr_opacity * G);
			if (alpha < 1.0f / 255.0f)
				continue;

			T = T / (1.f - alpha);
			const float dchannel_dcolor = alpha * T;
			float dL_dalpha = 0.0f;
			float dL_dcolor[C] = {0};
			float curr_dL_dinvdepth(0.f);
			glm::vec3 dL_dnormal(0.f);

			// TODO make that bool? No need for 2 values
			int2 clamped{0, 0};
			glm::vec2 weights(0.f);
			glm::vec<2, int> index, index_plus1, cutoff_index_start, cutoff_index_end;
			glm::vec3 colorSamples[4];
			const glm::vec3 &base_colour = curr_colour;
			glm::vec3 colour(base_colour);
			const glm::vec2 &uv = canonical2TexUV(
				curr_texture_resolution,
				curr_texelSize,
				curr_scale * (float)TEXTURE_GAUSSIAN_CUTOFF,
				intersection_point_axisaligned,
				cutoff_index_start,
				cutoff_index_end,
				clamped);
			// Interpolate
			if (!clamped.x && !clamped.y)
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
					colorSamples,
					false);
			}
			colour = glm::max(colour, glm::vec3(0.f));
			// Propagate gradients to per-Gaussian depths
			accum_depth_rec = last_alpha * last_invdepth + (1.f - last_alpha) * accum_depth_rec;
			last_invdepth = curr_invdepth;
			dL_dalpha += glm::dot((curr_invdepth - accum_depth_rec), dLdepth_dsplatted_invdepth);
			curr_dL_dinvdepth += dLdepth_dsplatted_invdepth * dchannel_dcolor;
#ifdef REND_NORMAL
			// Propagate gradients to per-Gaussian normals
			accum_normal_rec = last_alpha * last_normal + (1.f - last_alpha) * accum_normal_rec;
			last_normal = curr_normal;
			dL_dalpha += glm::dot((curr_normal - accum_normal_rec), dLnormal_dsplatted_normal);
			dL_dnormal += dLnormal_dsplatted_normal * dchannel_dcolor;
#endif
			// Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
			// pair).
			for (int ch = 0; ch < C; ch++)
			{
				float c{0.f};
				c = colour[ch];

				// Update last color (to be used in the next iteration)
				accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
				last_color[ch] = c;

				const float dL_dchannel = dL_dpixel[ch];
				dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;
				// Update the gradients w.r.t. color of the Gaussian.
				// Atomic, since this pixel is just one of potentially
				// many that were affected by this Gaussian.
				dL_dcolor[ch] = dL_dchannel * dchannel_dcolor;
				// if(c < 0.f)
				// {
				// 	dL_dcolor[ch] -= 1.f / n_primitives / (curr_texture_resolution.x * curr_texture_resolution.y);
				// }
				// else if(c > 1)
				// {
				// 	dL_dcolor[ch] += 1.f / n_primitives / (curr_texture_resolution.x * curr_texture_resolution.y);
				// }

				atomicAdd(&(dL_dcolors[global_id * C + ch]), dL_dcolor[ch]);
			}

			dL_dalpha *= T;
			// Update last alpha (to be used in the next iteration)
			last_alpha = alpha;

			// Account for fact that alpha also influences how much of
			// the background color is added if nothing left to blend
			float bg_dot_dpixel = 0.f;
			for (int i = 0; i < C; i++)
				bg_dot_dpixel += bg_color[i] * dL_dpixel[i];
			dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;

			// Helpful reusable temporary variables
			const float dL_dG = curr_opacity * dL_dalpha;
				// Backward for queried texture colour -> texture colour features
				// and queried texture colour -> uv coordinates
				// 2 because our texture is planar.
				glm::mat2x3 dcolor_duv(0.f);
				if (!clamped.x && !clamped.y)
				{
					getTextureColourBackward<C>(curr_texture_resolution,
												cutoff_index_start,
												cutoff_index_end,
												weights,
												index,
												index_plus1,
												colorSamples,
												dcolor_duv,
												dL_dcolor,
												// TODO this can overflow
												dL_dtextureFeatures + curr_texture_map_starting_offset * C);
				}

				const glm::vec2 dL_duv = glm::make_vec3(dL_dcolor) * dcolor_duv;

				glm::mat2 duv_dintersection_point_axisaligned(0.f);
				canonical2TexUVBackward(
					curr_texelSize,
					clamped,
					duv_dintersection_point_axisaligned);
				
				glm::vec2 dG_dcanonical(0.f);
				falloffCanonicalBackward(
					G,
					intersection_point_canonical,
					dG_dcanonical);

				// Separate the two contributions as scale needs only the one coming from falloff

				glm::vec2 curr_dL_dcanonical = dL_dG * dG_dcanonical;
				// Backward for Scale
				glm::mat2 curr_dcanonical_dscale(0.f);
				view2CanonicalScaleBackward(curr_scale, intersection_point_axisaligned, curr_dcanonical_dscale);
				atomicAddVector(dL_dscale + global_id, curr_dL_dcanonical * curr_dcanonical_dscale);

				const glm::vec2 curr_dcanonical_dunormalised = 1.f / curr_scale;
				glm::vec2 curr_dL_dintersection_point_axisaligned = dL_duv * duv_dintersection_point_axisaligned + curr_dL_dcanonical * curr_dcanonical_dunormalised;
				
				// Backward for Rotation
				glm::mat3 curr_dL_dR(0.f);
				const glm::vec3 intersection_point_world = inv_world2View * intersection_point_view;
				view2CanonicalRotationBackward(glm::vec3(curr_dL_dintersection_point_axisaligned, 0.f), intersection_point_world, curr_dL_dR);

				// xc = view2canonical xv
				const glm::mat3 &curr_dintersection_point_axisaligned_dintersection_point_view = inv_R * inv_world2View;
				
				// d(1/(z_intersection + z_mean))/d_z_intersection = - 1 / denominater**2
				const glm::vec3 curr_dinvdepth_dintersection_view = -glm::vec3(0.f, 0.f, 1.f / (curr_invdepth * curr_invdepth));
				
				// d(1/(z_intersection + z_mean))/d_z_mean = -1 / denominater**2
				const glm::vec3 &curr_dinvdepth_dmeanview = curr_dinvdepth_dintersection_view;

				// Backward for mean and rotation
				// Mean
				const glm::vec3 curr_dL_dintersection_point_view = glm::vec3(curr_dL_dintersection_point_axisaligned, 0.f) * curr_dintersection_point_axisaligned_dintersection_point_view + curr_dL_dinvdepth * curr_dinvdepth_dintersection_view;
				glm::vec3 curr_dL_dmeanview(0.f), curr_dL_dnormalview(dL_dnormal);
				intersectionPointViewBackward(curr_dL_dintersection_point_view, ray_direction_view, curr_normal, curr_p_view, curr_dL_dmeanview, curr_dL_dnormalview);
				// There is one more contribution from inverse depth
				curr_dL_dmeanview += curr_dL_dinvdepth * curr_dinvdepth_dmeanview;
				const glm::mat3 &curr_dmeanview_dmean = world2view;
				const glm::vec3 curr_dL_dmean = curr_dL_dmeanview * curr_dmeanview_dmean;
				atomicAddVector(dL_dmean3D + global_id, curr_dL_dmean);

				const glm::mat3 &curr_dnormalview_dnormal = world2view;
				const glm::vec3 curr_dL_dnormal = curr_dL_dnormalview * curr_dnormalview_dnormal;

				curr_dL_dR[2] += (float)curr_normal_sign * curr_dL_dnormal;
				// Rotation 2 quaternion
				glm::vec4 curr_dL_dquaternion(0.f);
				rotationQuaternionBackward(
					curr_dL_dR,
					rotations[global_id],
					curr_dL_dquaternion);

				atomicAddVector(dL_dquaternion + global_id, curr_dL_dquaternion);
			// Update gradients w.r.t. opacity of the Gaussian
			atomicAdd(&dL_dopacity[global_id], G * dL_dalpha);
		}
	}
}

void BACKWARD::preprocess(
	int P, const int D, int M,
	const int *radii,
	const float *shs,
	const CudaRasterizer::GeometryState &geomState,
	const CudaRasterizer::GeometryGradients &geomGrads,
	const float *viewmatrix,
	const float *projmatrix,
	const float2 focal,
	const float2 tan_fov,
	const glm::vec3 *campos,
	float *dL_dcolor,
	float *dL_dsh)
{
	// Propagate gradients for remaining steps: finish 3D mean gradients,
	// propagate color gradients to SH (if desireD), propagate 3D covariance
	// matrix gradients to scale and rotation.
	preprocessCUDA<NUM_CHANNELS><<<(P + 255) / 256, 256>>>(
		P, D, M,
		(float3 *)geomState.mean,
		radii,
		shs,
		geomState.buffer.clamped,
		projmatrix,
		campos,
		geomGrads.dL_dmean3D,
		dL_dcolor,
		dL_dsh);
}

void BACKWARD::render(
	const int n_primitives,
	const dim3 grid, const dim3 block,
	const uint32_t *point_list,
	int W, int H,
	const float *bg_color,
	const CudaRasterizer::TextureState& textureState,
	const CudaRasterizer::GeometryState& geomState,
	const CudaRasterizer::ImageState& imgState,
	const CudaRasterizer::TextureGradients& textureGrads,
	const CudaRasterizer::GeometryGradients& geomGrads,
	const float *colors,
	const float2 tan_fov,
	const float2 focal,
	const float *viewmatrix,
	float *dL_dcolors)
{
	renderCUDA<NUM_CHANNELS><<<grid, block>>>(
		n_primitives,
		imgState.ranges,
		point_list,
		W, H,
		bg_color,
		geomState.buffer.means2D,
		geomState.opacity,
		textureState.buffer.normal,
		(glm::vec3 *)colors,
		imgState.accum_alpha,
		imgState.n_contrib,
		textureState.textureMap,
		textureState.textureResolution,
		textureState.textureMapStartingOffset,
		textureState.texelSize,
		(glm::vec3 *)textureState.buffer.mean,
		textureState.buffer.view2canonical,
		textureState.buffer.normal_sign,
		focal,
		tan_fov,
		viewmatrix,
		geomState.scale,
		geomState.rotation,
		geomGrads.dL_dscale,
		geomGrads.dL_dquaternion,
		geomGrads.dL_dmean3D,
		geomGrads.dL_dpixels,
		geomGrads.dL_dout_features,
		geomGrads.dL_dopacity,
		dL_dcolors,
		textureGrads.dL_dtextureMap);
}