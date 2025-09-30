import torch

from diff_gaussian_rasterization_texture import GaussianRasterizer

from scene.cameras import Camera
import gaussian_renderer

class ErrorStats:
    def __init__(self, num_primitives):
        self.n_views = 0
        self.areas = torch.zeros((num_primitives, 1), device="cuda")
        self.contributions = torch.zeros((num_primitives, 1), device="cuda")
        self.errors = torch.zeros((num_primitives, 1), device="cuda")

    def accumulate_view_error_stats(
        self,
        view_error_stats: dict[str, torch.Tensor],
        view_camera: Camera,
    ):
        self.n_views += 1

        image_size = view_camera.image_width * view_camera.image_height

        # Add to accumulators
        view_areas = view_error_stats["n_pixels"] / image_size
        self.areas += view_areas
        self.contributions += view_error_stats["contributions"] / image_size
        self.errors += (view_error_stats["errors"] / image_size) * (view_error_stats["contributions"] / image_size)


    def normalize_error_stats(self, ) -> torch.Tensor:
        normalised_errors = self.errors / self.contributions
        normalised_errors.nan_to_num_(0, 0, 0)
        return normalised_errors

@torch.no_grad()
def compute_view_error_stats(gaussians, background, pipeline, camera):
    # TODO fix this abomination (some circular dependency issues that I didn't have 
    # the mental fortitude to solve at a specific point in time)
    from utils.loss_utils import ssim

    gt_image = camera.original_image

    render_pkg = gaussian_renderer.render_with_full_state(camera, gaussians, pipeline, background)

    rasterizer = GaussianRasterizer(render_pkg["raster_settings"])
    render_img = render_pkg["render"].clamp(0, 1)

    ssim_img = 1 - ssim(gt_image, render_img, aggregate=False).mean(dim=0, keepdim=True)
    l1_img = (gt_image - render_img).abs().mean(dim=0, keepdim=True)
    loss_img = 0.2 * ssim_img + 0.8 * l1_img
    view_error_stats = rasterizer.error_stats(
        render_pkg["means3D"],
        render_pkg["opacity"],
        render_pkg["shs"],
        render_pkg["colors_precomp"],
        render_pkg["scales"],
        render_pkg["rotations"],
        render_pkg["radii"],
        render_pkg["cov3Ds_precomp"],
        render_pkg["geom_buffer"],
        render_pkg["texture_buffer"],
        render_pkg["num_rendered"],
        render_pkg["raster_settings"].viewmatrix,
        render_pkg["binning_buffer"],
        render_pkg["img_buffer"],
        render_pkg["weight_precomputed"],
        loss_img
    )

    return view_error_stats


@torch.no_grad()
def compute_error_stats(gaussians, pipeline, background, cameras):
    error_stats = ErrorStats(gaussians.num_primitives)

    for camera in cameras:
        view_error_stats = compute_view_error_stats(
            gaussians, background, pipeline, camera
        )

        error_stats.accumulate_view_error_stats(
            view_error_stats, camera
        )
    error_stats.errors = error_stats.normalize_error_stats()
    return error_stats
