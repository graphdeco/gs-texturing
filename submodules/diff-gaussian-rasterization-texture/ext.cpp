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

#include <torch/extension.h>
#include "rasterize_points.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // m.def("rasterize_gaussians_variableSH_bands", &RasterizeGaussiansVariableSHBandsCUDA);
  m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);
  m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA);
  m.def("rasterize_gaussians_error_stats", &RasterizeGaussiansErrorStatsCUDA);
  m.def("mark_visible", &markVisible);
  m.def("copy_to_resized_tensor", &ResizeJaggedTensor);
  m.def("create_jagged_mask", &CreateJaggedMask);
  m.def("aggregate_projected_pixel_sizes", &calculatePixelSize);
  m.def("kmeans_cuda", &kmeans);
}