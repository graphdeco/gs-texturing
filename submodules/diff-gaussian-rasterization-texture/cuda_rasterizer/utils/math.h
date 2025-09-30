#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
//#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

__device__ glm::vec3 packed_self_outer_product(const glm::vec2 &a) {
    return {a.x * a.x, a.x * a.y, a.y * a.y};
}

__device__ float eval_packed_quadratic_form(const glm::vec3 &q, const glm::vec2 x) {
    auto res = q * packed_self_outer_product(x);

    return res.x + 2. * res.y + res.z;
}
