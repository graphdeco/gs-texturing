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

import os
from argparse import ArgumentParser

class SceneGroup:
    def __init__(self, scene_names, images_arg=""):
        self._scene_names = scene_names
        self._images_arg = images_arg
        self._source = None

mipnerf360_outdoor_scenegroup = SceneGroup(["bicycle", "flowers", "garden", "stump", "treehill"], "-i images_4")
mipnerf360_indoor_scenegroup = SceneGroup(["room", "counter", "kitchen", "bonsai"], "-i images_2")
tanks_and_temples_scenegroup = SceneGroup(["truck", "train"])
deep_blending_scenegroup = SceneGroup(["drjohnson", "playroom"])

scene_group_list = []
scene_group_list.append(mipnerf360_outdoor_scenegroup)
scene_group_list.append(mipnerf360_indoor_scenegroup)
scene_group_list.append(tanks_and_temples_scenegroup)
scene_group_list.append(deep_blending_scenegroup)

configuration = {}
# Paper experiments
configuration["final"] = ""
configuration["points40k"] = " --cap_max=40_000"
configuration["points80k"] = " --cap_max=80_000"
configuration["points120k"] = " --cap_max=120_000"
configuration["points160k"] = " --cap_max=160_000"

# Additional models
# More points
configuration["lower_splitting_threshold"] = " --splitting_threshold=16"
# More aggressive downsampling
configuration["higher_downscale_threshold"] = " --downscale_threshold=0.04"



all_scene_names = [scene for scene_group in scene_group_list for scene in scene_group._scene_names]

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--output_path", default="./eval", type=str)
parser.add_argument("--experiments", "-e", default=["final"], nargs="+", choices=list(configuration.keys()), type=str)
parser.add_argument("--scenes", "-s", default=all_scene_names, nargs="+", choices=all_scene_names, type=str)
parser.add_argument('--mipnerf360', "-m360", required=False, default="/data/graphdeco/user/ppapanto/m360", type=str)
parser.add_argument("--tanksandtemples", "-tat", required=False, default="/data/graphdeco/user/ppapanto/tat", type=str)
parser.add_argument("--deepblending", "-db", required=False, default="/data/graphdeco/user/ppapanto/db", type=str)
args = parser.parse_args()

mipnerf360_outdoor_scenegroup._source = args.mipnerf360
mipnerf360_indoor_scenegroup._source = args.mipnerf360
tanks_and_temples_scenegroup._source = args.tanksandtemples
deep_blending_scenegroup._source = args.deepblending

for scene_group in scene_group_list:
    for scene in scene_group._scene_names:
        if scene in args.scenes:
            for experiment in args.experiments:
                output_path = f"{args.output_path}/{scene}/{experiment}"
                if not args.skip_training:
                    common_args = " --quiet --eval --test_iterations -1 "
                    os.system(f"python train.py -s {scene_group._source}/{scene} {scene_group._images_arg} -m {output_path} {common_args} {configuration[experiment]}")

                if not args.skip_rendering:
                    common_args = f" --quiet --eval --skip_train "
                    os.system(f"python render.py --iteration 30000 -s {scene_group._source}/{scene} -m {output_path} {common_args}")

                if not args.skip_metrics:
                    os.system(f"python metrics.py -m {output_path}")
