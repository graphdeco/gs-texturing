#include "utils.h"
#include "auxiliary.h"



// Copies a source jagged tensor to a different sized target jagged tensor using masks to
// guide the assigments.
__global__ void CopyTensorCuda(
    const int n_iterations,
    const glm::vec3 *__restrict__ source_jagged_tensor,
    const int2 *__restrict__ source_resolution,
    const int *__restrict__ source_offset,
    const int *__restrict__ source_mask,
    const int *__restrict__ target_mask,
    glm::vec3 *__restrict__ target_jagged_tensor,
    int2 *__restrict__ target_resolution,
    int *__restrict__ target_offset,
    int2 *__restrict__ target_center_shift)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n_iterations) return;

    const int curr_source_idx = source_mask[idx];
    const int curr_target_idx = target_mask[idx];
    const glm::vec2 curr_new_center = {target_center_shift[idx].x, target_center_shift[idx].y};
    
    const int curr_source_offset = source_offset[curr_source_idx];
    const int curr_target_offset = target_offset[curr_target_idx];

    const glm::vec3* curr_source_jagged_tensor = source_jagged_tensor + curr_source_offset;
    glm::vec3* curr_target_jagged_tensor = target_jagged_tensor + curr_target_offset;

    // Since the tensors are given in height x width dimensions
    // resolution.x is up-down/rows
    // resolution.y is left-right/columns
    glm::vec2 curr_source_resolution = {source_resolution[curr_source_idx].x, source_resolution[curr_source_idx].y};
    glm::vec2 curr_target_resolution = {target_resolution[curr_target_idx].x, target_resolution[curr_target_idx].y};

    // Taking the element 0,0 of the source tensor as the origin,
    // we compute the translation of the target tensor
    // taking into account that the two tensors are centrally aligned (+ an optional offset)
    glm::vec<2, int> relative_shift(curr_source_resolution / 2.f - curr_target_resolution / 2.f - curr_new_center);

    int2 iteration_boundaries_start = {max(0, relative_shift.x), max(0, relative_shift.y)};
    int2 iteration_boundaries_end = {
        min(curr_source_resolution.x, relative_shift.x + curr_target_resolution.x),
        min(curr_source_resolution.y, relative_shift.y + curr_target_resolution.y)};

    // Copy the existing texture map so that it is centrally placed
    for (int i = iteration_boundaries_start.x; i < iteration_boundaries_end.x; ++i)
    {
        for (int j = iteration_boundaries_start.y; j < iteration_boundaries_end.y; ++j)
        {
                const int source_index_offset = i * curr_source_resolution.y + j;
                const int target_index_offset = (i - relative_shift.x) * curr_target_resolution.y + (j - relative_shift.y);
                curr_target_jagged_tensor[target_index_offset] = curr_source_jagged_tensor[source_index_offset];
        }
    }
}

void UTILS::CopyTensor(
    const int n_iterations,
    const float *source_jagged_tensor,
    const int2 *source_resolution,
    const int *source_offset,
    const int *source_mask,
    const int *target_mask,
    float *target_jagged_tensor,
    int2 *target_resolution,
    int *target_offset,
    int2 *target_center_shift)
{
	CopyTensorCuda<<<(n_iterations + 255) / 256, 256>>>(
        n_iterations,
        (glm::vec3 *)source_jagged_tensor,
        source_resolution,
        source_offset,
        source_mask,
        target_mask,
        (glm::vec3 *)target_jagged_tensor,
        target_resolution,
        target_offset,
        target_center_shift
    );
}

__global__ void CreateJaggedMaskCuda(
    const int n_primitives,
    const bool *__restrict__ mask,
    bool *__restrict__ jagged_tensor,
    int2 *__restrict__ resolution,
    int *__restrict__ offset
)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n_primitives) return;

    bool mask_value = mask[idx];
    
    bool* curr_jagged_tensor = jagged_tensor + offset[idx];
    int2 curr_resolution = resolution[idx];
    
    // Copy the value of the mask to all entries, using the given resolution
    for (int i = 0; i < curr_resolution.x; ++i)
    {
        for (int j = 0; j < curr_resolution.y; ++j)
        {
            curr_jagged_tensor[(i * curr_resolution.y) + j] = mask_value;
        }
    }
}

// Copies the values of the mask to all the corresponding entries
// of the jagged tensor
void UTILS::CreateJaggedMask(
    const int n_primitives,
    const bool *mask,
    bool *jagged_tensor,
    int2 *resolution,
    int *offset
)
{
    CreateJaggedMaskCuda<<<(n_primitives + 255) / 256, 256>>>(
        n_primitives,
        mask,
        jagged_tensor,
        resolution,
        offset
    );
}