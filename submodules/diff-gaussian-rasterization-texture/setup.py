#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

setup(
    name="diff_gaussian_rasterization_texture",
    packages=['diff_gaussian_rasterization_texture'],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization_texture._C",
            sources=[
            "cuda_rasterizer/rasterizer_impl.cu",
            "cuda_rasterizer/forward.cu",
            "cuda_rasterizer/backward.cu",
            "cuda_rasterizer/utils.cu",
            "cuda_rasterizer/error_stats.cu",
            "rasterize_points.cu",
            "ext.cpp"],
            extra_compile_args={
                "cxx": ['/w'],
                "nvcc": [
                "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/"),
                "--disable-warnings"
                ]})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
