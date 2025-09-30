#ifndef CUDA_RASTERIZER_UTILS_H_INCLUDED
#define CUDA_RASTERIZER_UTILS_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace UTILS
{
    void CopyTensor(
        const int n_iterations,
        const float *source_jagged_tensor,
        const int2 *source_resolution,
        const int *source_offset,
        const int *source_mask,
        const int *target_mask,
        float *target_jagged_tensor,
        int2 *target_resolution,
        int *target_offset,
        int2 *target_center_shift);

    void CreateJaggedMask(
        const int n_primitives,
        const bool *mask,
        bool *jagged_tensor,
        int2 *resolution,
        int *offset);
}

#endif