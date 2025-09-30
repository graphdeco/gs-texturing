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

import torch
import sys
from utils.general_utils import safe_state
from tqdm import tqdm
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, InitialisationParams
from controller import Trainer
from viewer import GaussianViewer, ViewerMode
from threading import Thread
import time
losses = ["l1_loss", "ssim_loss", "alpha_regul", "scale_regul_loss", "texture_regul", "colour_variance_loss", "sh_sparsity_loss", "total_loss", "iter_time"]

def training(dataset, opt, pipe, init_args, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, viewer_mode: ViewerMode):
    first_iter = 0

    trainer = Trainer(dataset, opt, pipe, init_args)

    trainer.training_setup()
    
    if checkpoint:
        trainer.restore(checkpoint)
        
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, trainer._opt_args.iterations), desc="Training progress")
    first_iter += 1

    # Initialize and start viewer in a separate thread
    mode = ViewerMode.LOCAL if viewer_mode == "local" else ViewerMode.SERVER
    viewer = GaussianViewer.from_gaussians(dataset, pipe, trainer.gmodel, mode, trainer.scene)

    if viewer_mode != "none":
        viewer_thd = Thread(target=viewer.run, daemon=True)
        viewer_thd.start()

    for iteration, viewpoint_cam in zip(range(first_iter, trainer._opt_args.iterations + 1), trainer.viewpoint_picker):
        # TODO: That's a stupid way to stall, fix it
        while not viewer.train:
            time.sleep(0.2)
    
        iter_start.record()

        if (iteration - 1) == debug_from:
            trainer._pipe_args.debug = True

        trainer.forward_pass(viewpoint_cam)
        trainer.compute_losses(viewpoint_cam.original_image.cuda())
        trainer.backward_pass()

        viewer.gaussian_lock.acquire()

        if iteration in saving_iterations:
            trainer.save_gmodel(iteration)

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * trainer.loss.detach().item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == trainer._opt_args.iterations:
                progress_bar.close()

            # Log
            trainer.log(iteration, iter_start.elapsed_time(iter_end), testing_iterations)
           
            if iteration == 500:
                trainer.activate_texture_training()

            # Densification
            if iteration >= trainer._opt_args.densify_from_iter and iteration % trainer._opt_args.densification_interval == 0:
                if trainer._opt_args.densify_from_iter <= iteration < trainer._opt_args.densify_until_iter:
                        trainer.compute_error()
                        trainer.primitive_management()
                else:
                    trainer.prune_primitives()

            if trainer._opt_args.densify_from_iter <= iteration <= min(trainer._opt_args.densify_until_iter,
                   (trainer._opt_args.iterations - 3000)) \
                    and iteration % (trainer._opt_args.densification_interval) == 0:
                trainer.adaptive_texel_size()

            # At fine-tuning stage perform only downscale of primitives 
            if iteration >= trainer._opt_args.densify_until_iter and iteration % 1000 == 0:
                trainer.downscale_primitives()

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                trainer.enable_next_sh_band()            

            trainer.step(iteration)
            viewer.gaussian_lock.release()


            if (iteration in checkpoint_iterations):
                trainer.capture(iteration)

    trainer.save_gmodel(iteration, True)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    ip = InitialisationParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--cull_SH", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument('--viewer_mode', choices=['local', 'server', 'none'], default='local')
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args),
    op.extract(args),
    pp.extract(args),
    ip.extract(args),
    args.test_iterations,
    args.save_iterations,
    args.checkpoint_iterations,
    args.start_checkpoint,
    args.debug_from,
    args.viewer_mode)

    # All done
    print("\nTraining complete.")
