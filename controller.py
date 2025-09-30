# This file is aimed to be a controller class/set of functions that will be the only one 
# containing scene and gaussians
# The fact that we have a gaussian object within a scene object doesn't make sense
# This can potentially store the training function, the logger, the viewer etc
import torch
from scene import Scene
from scene.cameras import Camera, MiniCam
from scene.gaussian_model import GaussianModel
from diff_gaussian_rasterization_texture._C import aggregate_projected_pixel_sizes
from error_stats import compute_error_stats
from arguments import ModelParams, PipelineParams, OptimizationParams, InitialisationParams
from utils.general_utils import get_expon_lr_func
from utils.general_utils import seq_random_permutation
from itertools import chain, repeat
from gaussian_renderer import render
from utils.loss_utils import l1_loss, ssim, evaluate_viewpoint
from error_stats import ErrorStats
from utils.jagged_tensor import JaggedTensor
import uuid
from utils.visualisation_utils import get_colormap
from argparse import Namespace
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False



import os

# TODO Move to another file
class OptimizerContainer:
    """Class encapsulating the optmizer and relevant functions"""

    def setup(self, opt_args: OptimizationParams, gmodel: GaussianModel, spatial_lr_scale: float):
        """Sets up object by storing all learning rates and creating the optmizer object"""
        self._xyz_lr = opt_args.position_lr_init
        self._f_dc_lr = opt_args.feature_lr
        self._f_rest_lr = opt_args.feature_lr / 20.0
        self._opacity_lr = opt_args.opacity_lr
        self._scaling_lr = opt_args.scaling_lr
        self._rotation_lr = opt_args.rotation_lr
        self._texture_map_lr = opt_args.texture_map_lr
        

        l = []        
        l.append({'params': [gmodel._xyz], 'lr': self._xyz_lr * spatial_lr_scale, "name": "xyz"})
        l.append({'params': [gmodel._features_dc], 'lr': self._f_dc_lr, "name": "f_dc"})
        l.append({'params': [gmodel._features_rest], 'lr': self._f_rest_lr, "name": "f_rest"})
        l.append({'params': [gmodel._opacity], 'lr': self._opacity_lr, "name": "opacity"})
        l.append({'params': [gmodel._scaling], 'lr': self._scaling_lr, "name": "scaling"})
        l.append({'params': [gmodel._rotation], 'lr': self._rotation_lr, "name": "rotation"})
        l.append({'params': [gmodel._texture_map._values], 'lr': self._texture_map_lr, "name": "texture_map"})

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=opt_args.position_lr_init * spatial_lr_scale,
                                                    lr_final=opt_args.position_lr_final * spatial_lr_scale,
                                                    lr_delay_mult=opt_args.position_lr_delay_mult,
                                                    max_steps=opt_args.position_lr_max_steps)

    def activate_training(self, param_group_name: str):
        """Activates training for a single parameters by setting the lr to the internally stored one"""

        for group in self.optimizer.param_groups:
            if group["name"] == param_group_name:
                group["lr"] = getattr(self, (f"_{param_group_name}_lr"))

    def deactivate_training(self, param_group_name: str):
        """Deactivates training for a single parameters by setting the lr to 0, storing the previous lr"""

        for group in self.optimizer.param_groups:
            if group["name"] == param_group_name:
                setattr(self, (f"_{param_group_name}_lr"), group["lr"])
                group["lr"] = 0

    def step(self):
        """Optiizer step and zero grad"""

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def step_schedulers(self, iteration: int):
        """Learning rate scheduling per step"""

        if isinstance(self.optimizer, torch.optim.Adam):
            for group in self.optimizer.param_groups:
                if group["name"] == "xyz":
                    lr = self.xyz_scheduler_args(iteration)
                    group['lr'] = lr

    def replace_tensor_to_optimizer(self, tensor: torch.Tensor, name: str, masks: tuple[torch.Tensor, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
        """Replaces a parameter group's tensor defined by name with a given tensor

        By default zeroes-out the internal optimiser state
        Two masks, one corresponding to the initial and one to the final state of the internal state
        can be given to retain the information
        """

        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if masks is not None:
                    initial_mask = masks[0]
                    exp_avg_state = stored_state["exp_avg"][initial_mask]
                    exp_avg_sq_state = stored_state["exp_avg_sq"][initial_mask]
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                if masks is not None:
                    final_mask = masks[1]
                    stored_state["exp_avg"][final_mask] = exp_avg_state
                    stored_state["exp_avg_sq"][final_mask] = exp_avg_sq_state
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = tensor.requires_grad_(True)
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask: torch.Tensor, texture_map_sizes: torch.Tensor) -> dict[str, torch.Tensor]:
        """Masks all parameter groups with the given mask"""

        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            grad = None
            if stored_state is not None:
                if group["name"] == "texture_map":
                    temp_jagged = JaggedTensor(texture_map_sizes, stored_state["exp_avg"])
                    mask = temp_jagged.create_jagged_mask(mask)
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                
                # Store the grads as they're gonna be zeroed-out
                if group["params"][0].grad is not None:
                    grad = group["params"][0].grad[mask]
                
                group["params"][0] = (group["params"][0][mask].requires_grad_(True))

                # Restore the grads
                if grad is not None:
                    group["params"][0].grad = grad
                self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = group["params"][0][mask].requires_grad_(True)
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def cat_tensors_to_optimizer(self, tensors_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Appends all internal parameter tensors with tensors coming from a dictionary
        
            In doing so, the internal optimizer state is set to 0
        """

        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            grad = None
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                
                # Store the grads as they're gonna be zeroed-out
                if group["params"][0].grad is not None:
                    grad = torch.cat((group["params"][0].grad, torch.zeros_like(extension_tensor)), dim=0)
                group["params"][0] = torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
                
                # Restore the grads
                if grad is not None:
                    group["params"][0].grad = grad
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                # Store the grads as they're gonna be zeroed-out
                if group["params"][0].grad is not None:
                    grad = torch.cat((group["params"][0].grad, torch.zeros_like(extension_tensor)), dim=0)
                group["params"][0] = torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
                
                # Restore the grads
                if grad is not None:
                    group["params"][0].grad = grad
                optimizable_tensors[group["name"]] = group["params"][0]
            torch.cuda.empty_cache()
        return optimizable_tensors

# I think I'd prefer that to be split into a Factory class (being the actual controller) and a Trainer class
# doing the dirty work of training, storing intermediate things and stuff
class Trainer:
    """Class that is responsible to perform training"""

    def __init__(self, model_args: ModelParams, opt_args: OptimizationParams, pipe_args: PipelineParams, init_args: InitialisationParams):
        """Initializes the object and its members"""

        # Store the entire arg objects/dicts to be passed as function arguments
        self._model_args = model_args
        self._opt_args = opt_args
        self._pipe_args = pipe_args
        self._init_args = init_args

        self.prepare_output_and_logger()
        
        self.scene: Scene = Scene(model_args, num_pts=model_args.cap_max if model_args.cap_max != -1 else 1_000_000)
        self.gmodel: GaussianModel = GaussianModel(model_args.sh_degree, opt_args.max_texture_resolution)
        self.optimizer: OptimizerContainer = OptimizerContainer()

        self.log_dict: dict[str, float] = {
            "l1_loss": 0,
            "ssim_loss": 0,
            "alpha_regul_loss": 0,
            "texture_regul_loss": 0,
            "iter_time": 0
        }

        # Extract some useful individual properties
        self.percent_dense: float = opt_args.percent_dense

        # Training stuff
        self.max_splitting_threshold = int(64)
        self.min_splitting_threshold = int(self._opt_args.splitting_threshold)
        self.curr_splitting_threshold = self.max_splitting_threshold

    def training_setup(self):
        """Sets up some variables that are needed for training"""

        self.gmodel.initialise_primitives(
            init_type=self._init_args.init_type,
            scene_info=self.scene.scene_info,
            scene_extent=self.scene.cameras_extent,
            samples_num=self._model_args.cap_max,
            ply_path=self._init_args.ply_path)
        
        with torch.no_grad():
            # Initialize pixel size, necessary for correct texture querying
            self.gmodel._pixel_size = Controller.aggregate_projected_pixel_sizes_cuda(self.gmodel, self.scene.getTrainCameras())
            self.gmodel.initialise_texel_pixel_ratio(initial_texture_resolution = torch.tensor([8,8], device="cuda"))
        
        # Generator that returns a camera, ensuring a full, randomised access to the dataset at each epoch
        # Credits to Petros
        self.viewpoint_picker: chain[Camera] = chain.from_iterable(map(seq_random_permutation, repeat(self.scene.getTrainCameras())))
    
        # Choose background color
        if self._opt_args.random_background:
            self.bg_color = torch.rand((3), dtype=torch.float32, device="cuda")
        else:
            self.bg_color = torch.tensor([1, 1, 1] if self._model_args.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")

        # Holds information for newly added, upsampled and downsampled primitives
        self.gmodel.error_stats = ErrorStats(self.gmodel.num_primitives)

        self.optimizer.setup(self._opt_args, self.gmodel, self.scene.cameras_extent)
        self.optimizer.deactivate_training("texture_map")

    def pick_camera(self) -> Camera:
        """Returns a camera. The dataset is traversed in full and in a random order at every epoch"""

        return next(self.viewpoint_picker)
    
    def forward_pass(self, viewpoint: Camera):
        """Runs the model's forward pass, storing the necesasry information to compute the losses"""

        render_pkg = Controller.render_gmodel(self.gmodel, self._pipe_args, self.bg_color, viewpoint)
        self.image: torch.Tensor = render_pkg["render"]
        self.visibility_filter: torch.Tensor = render_pkg["visibility_filter"]
        self.out_weight: torch.Tensor = render_pkg["out_weight"]

    def compute_losses(self, gt_image: torch.Tensor):
        """Compute the different loss functions and the total loss."""

        # Per gaussian losses
        if self._opt_args.lambda_alpha_regul == 0:
            Lalpha_regul = torch.tensor([0.], device="cuda")
        else:
            Lalpha_regul = (self.gmodel.get_opacity * self.visibility_filter.view(-1, 1)).mean()

        Ltexture_regul = (self.gmodel.texture_offsets.abs() * self.gmodel._texture_map.create_jagged_mask(self.visibility_filter).view(-1, 1)).sum()
        # Ltexture_regul = (self.gmodel.texture_offsets.abs() * (1-self.out_weight.detach()).repeat_interleave(repeats=torch.prod(self.gmodel._texture_map._sizes, dim=1).int()).view(-1, 1) * self.gmodel._texture_map.create_jagged_mask(self.visibility_filter).view(-1, 1)).sum()

        Ll1 = l1_loss(self.image, gt_image)
        ssim_loss = 1.0 - ssim(self.image, gt_image)

        self.loss = Ll1 * (1.0 - self._opt_args.lambda_dssim) \
               + ssim_loss * self._opt_args.lambda_dssim \
               + Lalpha_regul * self._opt_args.lambda_alpha_regul \
               + Ltexture_regul * self._opt_args.lambda_texture_regul

        self.log_dict["l1_loss"] += Ll1.detach().item()
        self.log_dict["ssim_loss"] += ssim_loss.detach().item()
        self.log_dict["alpha_regul_loss"] += Lalpha_regul.detach().item()
        self.log_dict["texture_regul_loss"] += Ltexture_regul.detach().item()

    def backward_pass(self):
        """Calls the backward function on the loss tensor"""

        self.loss.backward()

    def compute_error(self):
        """Runs the per primitive, contribution weighted loss computation"""

        self.gmodel.texture_map_resize(self.optimizer)
        Controller.compute_error(self.gmodel, self.scene, self._pipe_args, self.bg_color)

    def primitive_management(self):
        """Performs densification and pruning.
        For both axes, it calls the split primitives function for both axes and prunes"""

        n_added_primitives = 0
        for direction_idx in [0, 1]:
            n_primitives_before = self.gmodel.num_primitives
            prune_mask = self.gmodel.split_textured_primitives(
                self.optimizer,
                torch.tensor([round(self.curr_splitting_threshold), round(self.curr_splitting_threshold)], dtype=torch.int32, device="cuda"),
                direction_idx,
                percentile=self._opt_args.adaptive_texelsize_percentile,
                max_points=self._model_args.cap_max)
            n_primitives_after = self.gmodel.num_primitives
            n_added_primitives += n_primitives_after - n_primitives_before
            self.prune_primitives(10/255, prune_mask)
        
        # Zero error for the newly added primitives
        self.gmodel.error_stats.errors[-n_added_primitives:] = 0
        self.gmodel.texture_map_resize(self.optimizer)


    @torch.no_grad()
    def prune_primitives(self, min_opacity: float = 1/255, prune_mask: torch.Tensor | None = None):
        """"Calls the Gaussian Model's prune function and fixed the pixel sizes afterwards"""

        self.gmodel.prune(self.optimizer, min_opacity, self.scene.cameras_extent, prune_mask)
        self.gmodel._pixel_size = Controller.aggregate_projected_pixel_sizes_cuda(self.gmodel, self.scene.getTrainCameras())


    @torch.no_grad()
    def adaptive_texel_size(self):
        """Adaptive texel size routine. It calls the upscale and downscale functions,
        effectively changing the texel size of the primitives"""

        self.gmodel._pixel_size = Controller.aggregate_projected_pixel_sizes_cuda(self.gmodel, self.scene.getTrainCameras())
        percentile = self._opt_args.adaptive_texelsize_percentile
        errors: torch.Tensor = self.gmodel.error_stats.errors * (self.gmodel._texel_pixel_ratio > 1)
        mask = (errors > max(errors.quantile(percentile).item(), 1e-6)).view(-1)
        self.upscale_primitives(mask)
        self.downscale_primitives(~mask)

            
    def upscale_primitives(self, mask: torch.Tensor):
        """Function that calls the upscaling routine for all primitives"""

        self.gmodel.increase_texture_resolution(self.optimizer, mask)
        self.gmodel.texture_map_resize(self.optimizer)
    
    def downscale_primitives(self, mask: torch.Tensor | None = None):
        """Function that calls the downscaling routine for all primitives"""

        if mask is None:
            mask = torch.ones(self.gmodel.num_primitives, dtype=torch.bool, device="cuda")

        # Prevent downscale of primitives that have a low texture resolution already
        inv_low_texres_mask = (self.gmodel._calculate_active_texture_resolution(powers_of_two=True) > torch.tensor([2,2], device="cuda")).any(dim=1)

        self.gmodel.decrease_texture_resolution(
            self.optimizer,
            torch.logical_and(mask, inv_low_texres_mask),
            downscale_threshold=self._opt_args.downscale_threshold
        )
        self.gmodel.texture_map_resize(self.optimizer)

    
    def step(self, iteration: int):
        """End of iteration step procedure, including optimizer step and texture resize"""
        if iteration < self._opt_args.iterations and iteration % self._opt_args.batch_size == 0:
            self.optimizer_step(iteration)

            # At the first iteration, calculate the sizes
            # (some issue with the optimizer prevents us to do it before)
            if iteration == 1 or iteration % 100 == 0:
                self.gmodel.texture_map_resize(self.optimizer)

            # Update the splitting threshold
            self.curr_splitting_threshold = max(self.curr_splitting_threshold-(self.max_splitting_threshold-self.min_splitting_threshold) / 7000, self.min_splitting_threshold)


    def optimizer_step(self, iteration: int):
        """Calls optimizer and schedulers step"""

        self.optimizer.step()
        self.optimizer.step_schedulers(iteration)

    def save_gmodel(self, iteration: int, quantize: bool = False):
        """Saves the GaussianModel"""
        
        self.prune_primitives()

        Controller.save_gmodel(
            self.gmodel,
            self._model_args.model_path,
            iteration,
            self.optimizer,
            quantize
            )

    def enable_next_sh_band(self):
        """Wrapper for the oneupSHdegree function"""

        self.gmodel.oneupSHdegree()

    def activate_texture_training(self):
        """Activates training for texture maps"""

        self.optimizer.activate_training("texture_map")
        self.gmodel.initialise_texel_pixel_ratio(initial_texture_resolution = torch.tensor([8,8], device="cuda"))

    def log(self, iteration: int, elapsed_time: float, testing_iterations: list[int]):
        """If tensorboard is installed and enabled, logs various information to the writer."""

        if self.tb_writer:
            self.log_dict["iter_time"] += elapsed_time
            if iteration % self._opt_args.densification_interval == 0:
                total_loss = 0
                for name in self.log_dict:
                    total_loss += self.log_dict[name]
                    self.tb_writer.add_scalar(f'train_loss_patches/{name}', self.log_dict[name]/self._opt_args.densification_interval, iteration)
                    self.log_dict[name] = 0
                self.tb_writer.add_scalar('train_loss_patches/total_loss', total_loss/self._opt_args.densification_interval, iteration)
                self.tb_writer.add_scalar('memory_stats/active_texels', torch.prod(self.gmodel._calculate_active_texture_resolution(), dim=1).sum(), iteration)
                self.tb_writer.add_scalar('memory_stats/active_memory', (self.gmodel.num_primitives * 59 * 4 + torch.prod(self.gmodel._calculate_active_texture_resolution(), dim=1).sum().item() * 3)/1024**2, iteration)
                self.tb_writer.add_scalar('total_points', self.gmodel.num_primitives, iteration)

            # Report test and samples of training set
            if iteration in testing_iterations:
                torch.cuda.empty_cache()
                validation_configs = ({'name': 'test', 'cameras' : self.scene.getTestCameras()}, 
                                    {'name': 'train', 'cameras' : [self.scene.getTrainCameras()[idx % len(self.scene.getTrainCameras())] for idx in range(5, 30, 5)]})

                for config in validation_configs:
                    if config['cameras'] and len(config['cameras']) > 0:
                        l1_test = 0.0
                        ssim_test = 0.0
                        psnr_test = 0.0
                        for idx, viewpoint in enumerate(config['cameras']):
                            (l1_score,
                            ssim_score,
                            psnr_score,
                            image,
                            gt_image) = evaluate_viewpoint(self.gmodel, viewpoint, self._pipe_args, self.bg_color)
                            if self.tb_writer and (idx < 5) and (iteration in [7000, 25000, 30000]):
                                self.tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                                self.tb_writer.add_images(config['name'] + "_view_{}/l1_loss".format(viewpoint.image_name), get_colormap()[:, ((gt_image-image).cpu().abs().mean(dim=0)*255).int().clamp(0,255)][None], global_step=iteration)
                                if iteration == testing_iterations[0]:
                                    self.tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                            l1_test += l1_score
                            ssim_test += ssim_score
                            psnr_test += psnr_score
                        psnr_test /= len(config['cameras'])
                        ssim_test /= len(config['cameras'])
                        l1_test /= len(config['cameras'])
                        print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                        if self.tb_writer:
                            self.tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                            self.tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim_loss', ssim_test, iteration)
                            self.tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

                if self.tb_writer:
                    self.tb_writer.add_histogram("scene/opacity_histogram", self.gmodel.get_opacity, iteration)
                torch.cuda.empty_cache()

    def prepare_output_and_logger(self):
        """Sets up the logger and the outuput folder, logging the command line arguments on the way"""

        if not self._model_args.model_path:
            if os.getenv('OAR_JOB_ID'):
                unique_str = os.getenv('OAR_JOB_ID', 'experiment_')
            else:
                unique_str = str(uuid.uuid4())
            self._model_args.model_path = os.path.join("./output/", unique_str[0:10])
            
        # Set up output folder
        print("Output folder: {}".format(self._model_args.model_path))
        os.makedirs(self._model_args.model_path, exist_ok = True)
        with open(os.path.join(self._model_args.model_path, "cfg_args"), 'w') as cfg_log_f:
            cfg_log_f.write(str(Namespace(**vars(self._model_args), **vars(self._opt_args), **vars(self._pipe_args), **vars(self._init_args))))

        # Create Tensorboard writer
        self.tb_writer = None
        if TENSORBOARD_FOUND:
            self.tb_writer = SummaryWriter(self._model_args.model_path)
        else:
            print("Tensorboard not available: not logging progress")

    def capture(self, iteration: int):
        """Captures the current state of training by storing the gmodel and the optimizer"""

        print("\n[ITER {}] Saving Checkpoint".format(iteration))
        torch.save((self.gmodel.active_sh_degree,
                    self.gmodel._xyz,
                    self.gmodel._features_dc,
                    self.gmodel._features_rest,
                    self.gmodel._scaling,
                    self.gmodel._rotation,
                    self.gmodel._opacity,
                    self.gmodel._texture_map,
                    self.gmodel.texel_size,
                    self.optimizer.optimizer.state_dict(),
                    iteration),
                    self._model_args.model_path + "/chkpnt" + str(iteration) + ".pth")    

    def restore(self, path: str) -> int:
        """Restores the trainer to the checkpoint's state"""

        (self.gmodel.active_sh_degree,
        self.gmodel._xyz,
        self.gmodel._features_dc,
        self.gmodel._features_rest,
        self.gmodel._scaling,
        self.gmodel._rotation,
        self.gmodel._opacity,
        self.gmodel._texture_map,
        self.gmodel._texel_size,
        opt_dict,
        iteration) = torch.load(path)

        self.optimizer.optimizer.load_state_dict(opt_dict)
        self.gmodel._pixel_size = Controller.aggregate_projected_pixel_sizes_cuda(self.gmodel, self.scene.getTrainCameras())
        self.gmodel.initialise_texel_pixel_ratio(initial_texture_resolution = torch.tensor([8,8], device="cuda"))
        
        return iteration


# TODO Move to another file, rename
class Controller:
    """Factory class that rests at the core of the system, managing all different parts of it
    and acting as the mediator between the two most important ones; the scene and the gaussian model
    """
    
    @staticmethod
    def compute_projected_pixel_size(gmodel: GaussianModel, camera: Camera, initial_value: float = 10000) -> torch.Tensor:
        pixel_sizes_world = initial_value * torch.ones_like(gmodel.get_opacity).detach()

        w2ndc_transform = camera.full_proj_transform
        ndc_centers_hom = torch.matmul(torch.cat((gmodel.get_xyz, torch.ones((gmodel.num_primitives, 1), device="cuda")), dim=1).unsqueeze(1), w2ndc_transform.unsqueeze(0)).squeeze()
        ndc_centers_hom /= ndc_centers_hom.clone()[:, -1:]

        depths = ndc_centers_hom[:, 2]
        # Frustum culling. This should utilise visibility filter/radii as in cuda
        mask = torch.logical_and(
            torch.logical_and(
                torch.logical_and(ndc_centers_hom[:, 0:1] <= 1, ndc_centers_hom[:, 0:1] >= -1),
                torch.logical_and(ndc_centers_hom[:, 1:2] <= 1, ndc_centers_hom[:, 1:2] >= -1)),
            torch.logical_and(ndc_centers_hom[:, 2:3] <= 1, ndc_centers_hom[:, 2:3] >= 0)).squeeze()
        
        p_hom = torch.zeros_like(ndc_centers_hom)
        p_hom[:, 0 if camera.image_width > camera.image_height else 1] = min(2/camera.image_width, 2/camera.image_height)
        p_hom[:, 2] = depths
        p_hom[:, 3] = torch.ones(gmodel.num_primitives, device="cuda")
        
        p_hom_zero = torch.zeros_like(ndc_centers_hom)
        p_hom_zero[:, 2] = depths
        p_hom_zero[:, 3] = torch.ones(gmodel.num_primitives, device="cuda")
        
        # NDC [-1, 1] x [-1, 1] -> Pixel space [0, W] x [0, H]
        # [x, y, depth, 1] [x', y, depth, 1] -> [x_proj, y_proj, depth_proj, 1] - [x'_proj, y_proj, depth_proj, 1] -> [dx_proj, 0, 0, 1]
        # [dx, 0, depth, 1] [0, dy, depth, 1]

        p_hom[mask] = p_hom[mask] @ w2ndc_transform.inverse().unsqueeze(0)
        p_hom[mask] /= p_hom[mask][:, -1:]

        p_hom_zero[mask] = p_hom_zero[mask] @ w2ndc_transform.inverse().unsqueeze(0)
        p_hom_zero[mask] /= p_hom_zero[mask][:, -1:]

        pixel_sizes_world[mask] = torch.norm((p_hom[mask] - p_hom_zero[mask])[:, :3], dim=1, keepdim=True)
        return pixel_sizes_world

    @staticmethod
    def aggregate_projected_pixel_sizes_python(gmodel: GaussianModel, cameras: list[Camera], initial_value: float = 10000., aggregate_max: bool = False) -> torch.Tensor:
        # Initialise to a very high number
        pixel_sizes_world = initial_value * torch.ones_like(gmodel.get_opacity).detach()
        for camera in cameras:
            if aggregate_max:
                pixel_sizes_world = torch.max(pixel_sizes_world, Controller.compute_projected_pixel_size(gmodel, camera, initial_value))
            else:
                pixel_sizes_world = torch.min(pixel_sizes_world, Controller.compute_projected_pixel_size(gmodel, camera, initial_value))

        # Unseen primitives will have the initial, high value
        # Here we choose to set that to the next highest value
        if pixel_sizes_world.max() >= initial_value:
            initial_values_mask = pixel_sizes_world == pixel_sizes_world.max()
            second_highest_value = pixel_sizes_world[~initial_values_mask].max()
            pixel_sizes_world[initial_values_mask] = second_highest_value
        return pixel_sizes_world

    @staticmethod
    def compute_projected_pixel_size_cuda(gmodel: GaussianModel, camera: Camera) -> torch.Tensor:
        return aggregate_projected_pixel_sizes(
            torch.stack([camera.full_proj_transform]),
            torch.stack([camera.inverse_full_proj_transform]),
            gmodel._xyz,
            torch.tensor([camera.image_height], dtype=torch.int32, device="cuda"),
            torch.tensor([camera.image_width], dtype=torch.int32, device="cuda"))

    @staticmethod
    def aggregate_projected_pixel_sizes_cuda(gmodel: GaussianModel, cameras: list[Camera], initial_value: float = 10000., aggregate_max: bool = False) -> torch.Tensor:
        pixel_sizes = aggregate_projected_pixel_sizes(
            torch.stack([camera.full_proj_transform for camera in cameras]),
            torch.stack([camera.inverse_full_proj_transform for camera in cameras]),
            gmodel._xyz,
            torch.tensor([camera.image_height for camera in cameras], dtype=torch.int32, device="cuda"),
            torch.tensor([camera.image_width for camera in cameras], dtype=torch.int32, device="cuda"),
            initial_value,
            aggregate_max)
        mask = pixel_sizes == initial_value
        pixel_sizes[mask] = pixel_sizes[~mask].max()
        return pixel_sizes
        

    @staticmethod
    def compute_error(gmodel: GaussianModel, scene: Scene, pipe, background):
        gmodel.error_stats = compute_error_stats(gmodel, pipe, background, scene.getTrainCameras())

    @staticmethod
    def render_gmodel(
        gmodel: GaussianModel,
        pipe_args: PipelineParams,
        bg_color: torch.Tensor,
        viewpoint: Camera | MiniCam,
        nn_interpolation: bool = False,
        colour_type: str = "full",
        scaling_modifier: float = 1.0):
        """Renders the representation from the given viewpoint.
        Returns a dictionary containing primary and secondary rendering products (e.g. RGB image, visibility_filter)"""

        return render(viewpoint, gmodel, pipe_args, bg_color, scaling_modifier, texture_debug_view=nn_interpolation, colour_type=colour_type)
    
    @staticmethod
    def render_image(
        gmodel: GaussianModel,
        pipe_args: PipelineParams,
        bg_color: torch.Tensor,
        viewpoint: Camera | MiniCam) -> torch.Tensor:
        """Renders the representation from the given viewpoint and returns the resulting RGB image."""
        return Controller.render_gmodel(gmodel, pipe_args, bg_color, viewpoint)["render"]
    
    @staticmethod
    # TODO Should not use optimizer, get rid of it
    def save_gmodel(gmodel: GaussianModel, model_path: str, iteration: int, optimizer: OptimizerContainer, quantize: bool):
        point_cloud_path = os.path.join(model_path, "point_cloud/iteration_{}".format(iteration))
        gmodel.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        gmodel.save_texture_maps(point_cloud_path, optimizer, quantize)

    @staticmethod
    def load_gmodel(path: str, quantize: bool = False, sh_degree: int = 3, max_texture_resolution: int = 256) -> GaussianModel:
        gmodel: GaussianModel = GaussianModel(sh_degree, max_texture_resolution)
        gmodel.load_ply(os.path.join(path, "point_cloud.ply"))
        gmodel.load_texture_maps(path, quantize)

        return gmodel

    @staticmethod
    def load_scene(model_args: ModelParams) -> Scene:
        scene: Scene = Scene(model_args, num_pts=model_args.cap_max if model_args.cap_max != -1 else 1_000_000)
        return scene
