import os
from OpenGL.GL import *
from threading import Lock
from argparse import ArgumentParser
from imgui_bundle import imgui_ctx, imgui, ImVec2
from graphdecoviewer import Viewer
from graphdecoviewer.types import ViewerMode
from graphdecoviewer.widgets import Widget
from graphdecoviewer.widgets.image import TorchImage
from graphdecoviewer.widgets.cameras.fps import FPSCamera
from graphdecoviewer.widgets.monitor import PerformanceMonitor
from controller import Controller
from scene import Scene
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from arguments import ModelParams, PipelineParams
from torchvision.transforms import Resize


class GaussianModelStatistics(Widget):
    def __init__(self, mode: ViewerMode):
        """
        Displays the statistics of Gaussians like number of primitives
        """
        super().__init__(mode)


    def show_gui(self, gmodel: GaussianModel):
        imgui.text(f"Number of primitives {gmodel.num_primitives}")

class CameraInformation(Widget):
    def __init__(self, mode: ViewerMode):
        """
        Displays the statistics of Gaussians like number of primitives
        """
        super().__init__(mode)
        self.image_display = TorchImage(mode)
        self.camera: Camera
        self.display_mode: str = "rgb"
        self.display_size: ImVec2 = ImVec2(500, 500)

    def setup(self):
        self.image_display.setup()

    def step(self, camera: Camera, is_test_view: bool):
        self.camera = camera
        self.is_test_view = is_test_view

        # Calculate information to resize the image to fit the window
        aspect_ratio = self.camera.original_image.shape[2] / self.camera.original_image.shape[1]
        size_w, size_h = self.display_size[0], self.display_size[1]
        try:
            display_aspect_ratio = size_w / size_h
        except ZeroDivisionError:
            display_aspect_ratio = 1

        if display_aspect_ratio < aspect_ratio:
            size_h = size_w / aspect_ratio
        else:
            size_w = size_h * aspect_ratio

        size_h = max(int(size_h), 16)
        size_w = max(int(size_w), 16)

        match self.display_mode:
            case "rgb":
                if size_h != self.camera.original_image.shape[0] or size_w != self.camera.original_image.shape[1]:
                    img = Resize([size_h, size_w])(self.camera.original_image)
                else:
                    img = self.camera.original_image
                self.image_display.step(img.permute(1,2,0))

    def show_gui(self, size: ImVec2 = ImVec2(500, 500)):
        self.display_size = size
        self.image_display.show_gui()

class SceneControls(Widget):
    def __init__(self, mode: ViewerMode, scene: Scene):
        """
        Displays controls and information related to the loaded scene
        """
        super().__init__(mode)
        self.scene: Scene
        self.cameras_list: list[Camera]
        self.selected_camera_id: int = 0

        self.camera_info = CameraInformation(mode)
    
        self.scene = scene
        train_cameras_set = set(self.scene.getTrainCameras())
        test_cameras_set = set(self.scene.getTestCameras())
        
        self.all_cameras_list = self.scene.getTrainCameras() + self.scene.getTestCameras()
        self.all_cameras_list.sort(key = lambda cam: cam.image_name)

        self.train_ids = []
        self.test_ids = []

        for idx, cam in enumerate(self.all_cameras_list):
            if cam in train_cameras_set:
                self.train_ids.append(idx)
            elif cam in test_cameras_set:
                self.test_ids.append(idx)
            else:
                raise Exception("shouldn't reach here")
        
        self.camera_info.camera = self.all_cameras_list[self.selected_camera_id]
        self.snap: bool = False

    def setup(self):
        self.camera_info.setup()

    def step(self):
        self.camera_info.step(self.all_cameras_list[self.selected_camera_id], self.selected_camera_id in self.test_ids)

    def show_gui(self):
        imgui.separator_text("Scene Information")
        imgui.text(f"#Views: {len(self.all_cameras_list)}")

        imgui.separator_text("Controls")
        _, new_selected_camera_id = imgui.input_int("Select Camera", self.selected_camera_id, 1, 8)
        
        if new_selected_camera_id >= len(self.all_cameras_list):
            self.selected_camera_id = 0
        elif new_selected_camera_id < 0:
            self.selected_camera_id = len(self.all_cameras_list) - 1
        else:
            self.selected_camera_id = new_selected_camera_id

        _, self.snap = imgui.checkbox("Snap", self.snap)
        
        with imgui_ctx.begin(f"Selected Camera"):
            self.camera_info.show_gui(imgui.get_content_region_avail())

class Dummy(object):
    pass

class GaussianViewer(Viewer):
    def __init__(self, mode: ViewerMode):
        super().__init__(mode)
        self.window_title = "Gaussian Viewer"
        self.gaussian_lock = Lock()
        self.gmodel: GaussianModel
        self.scene: Scene
        self.train: bool = False

    def import_server_modules(self):
        global torch
        import torch

        global GaussianModel
        from scene.gaussian_model import GaussianModel

        global PipelineParams, ModelParams
        from arguments import PipelineParams, ModelParams

        global MiniCam
        from scene.cameras import MiniCam

        global render
        from gaussian_renderer import render

    @classmethod
    def from_ply(cls, dataset: ModelParams, pipe: PipelineParams, iter, mode: ViewerMode):
        viewer = cls(mode)

        # Read configuration

        ply_path = os.path.join(dataset.model_path, "point_cloud", f"iteration_{iter}")
        viewer.gmodel = Controller.load_gmodel(ply_path, sh_degree=dataset.sh_degree)
        viewer.gmodel._pixel_size = torch.ones_like(viewer.gmodel._texel_size)
        viewer.gmodel.initialise_texel_pixel_ratio(torch.tensor([2,2], dtype=torch.int32, device="cuda"))

        if "source_path" in args:
            viewer.scene = Controller.load_scene(dataset)

        viewer.dataset = dataset
        viewer.pipe = pipe

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        viewer.background = background
        return viewer
    
    @classmethod
    def from_gaussians(cls, dataset, pipe, gmodel: GaussianModel, mode: ViewerMode, scene: None | Scene = None):
        viewer = cls(mode)
        viewer.dataset = dataset
        viewer.pipe = pipe
        viewer.gmodel = gmodel

        if scene:
            viewer.scene = scene

        viewer.background = torch.tensor([0,0,0], dtype=torch.float32, device="cuda")
        return viewer

    def create_widgets(self):
        self.camera = FPSCamera(self.mode, 1297, 840, 47, 0.001, 100)
        self.point_view = TorchImage(self.mode)
        self.monitor = PerformanceMonitor(self.mode, ["Render"], add_other=False)
        self.gmodel_stats = GaussianModelStatistics(self.mode)

        if hasattr(self, 'scene'):
            self.scene_controls = SceneControls(self.mode, self.scene)

        # Render modes
        self.render_modes = ["Splats"]
        self.render_mode = 0

        # Render settings
        self.scaling_modifier = 1.0
        self.nn_interpolation: bool = False
        self.show_textures: bool = True

    def step(self):
        camera = self.camera
        world_to_view = torch.from_numpy(camera.to_camera).cuda().transpose(0, 1)
        full_proj_transform = torch.from_numpy(camera.full_projection).cuda().transpose(0, 1)
        camera = MiniCam(camera.res_x, camera.res_y, camera.fov_y, camera.fov_x, camera.z_near, camera.z_far, world_to_view, full_proj_transform)

        if self.render_mode == 0:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with torch.no_grad():
                with self.gaussian_lock:
                    net_image = Controller.render_gmodel(self.gmodel,
                    self.pipe,
                    self.background,
                    camera,
                    nn_interpolation=self.nn_interpolation,
                    colour_type="full" if self.show_textures else "base",
                    scaling_modifier=self.scaling_modifier)["render"]
                net_image = net_image.permute(1, 2, 0)
            end.record()
            end.synchronize()
            self.point_view.step(net_image)
            render_time = start.elapsed_time(end)
        
        if hasattr(self, 'scene_controls'):
            # TODO: put in a function
            if self.scene_controls.snap:
                selected_camera = self.scene_controls.all_cameras_list[self.scene_controls.selected_camera_id]
                self.camera.origin = selected_camera.camera_center.cpu().numpy()
                v2w = selected_camera.world_view_transform.inverse().cpu().numpy()
                self.camera.forward = v2w[2, :3]
                self.camera.up = -v2w[1, :3]
                self.camera.right = v2w[0, :3]
                self.camera.fov_x = selected_camera.FoVx
                self.camera.fov_y = selected_camera.FoVy
                
            self.scene_controls.step()
        
        self.monitor.step([render_time])
    
    def show_gui(self):
        with imgui_ctx.begin(f"Point View Settings"):
            _, self.render_mode = imgui.list_box("Render Mode", self.render_mode, self.render_modes)

            imgui.separator_text("Render Settings")
            if self.render_mode == 0:
                _, self.train = imgui.checkbox("Train", self.train)
                _, self.scaling_modifier = imgui.drag_float("Scaling Factor", self.scaling_modifier, v_min=0, v_max=1, v_speed=0.01)
                _, self.nn_interpolation = imgui.checkbox("NN Interpolation", self.nn_interpolation)
                _, self.show_textures = imgui.checkbox("Textures", self.show_textures)

            imgui.separator_text("Camera Settings")
            self.camera.show_gui()

        with imgui_ctx.begin("Point View"):
            if self.render_mode == 0:
                self.point_view.show_gui()

            if imgui.is_item_hovered():
                self.camera.process_mouse_input()
            
            if imgui.is_item_focused() or imgui.is_item_hovered():
                self.camera.process_keyboard_input()
        
        with imgui_ctx.begin("Performance"):
            self.monitor.show_gui()
        
        with imgui_ctx.begin(f"GModel Stats"):
            self.gmodel_stats.show_gui(self.gmodel)
        
        if hasattr(self, "scene_controls"):
            with imgui_ctx.begin(f"Scene Controls"):
                self.scene_controls.show_gui()

    def client_send(self):
        return None, {
            "scaling_modifier": self.scaling_modifier,
            "render_mode": self.render_mode
        }
    
    def server_recv(self, _, text):
        self.scaling_modifier = text["scaling_modifier"]
        self.render_mode = text["render_mode"]
    
if __name__ == "__main__":
    parser = ArgumentParser()
    lp = ModelParams(parser)
    pp = PipelineParams(parser)

    subparsers = parser.add_subparsers(title="mode", dest="mode", required=True)
    local = subparsers.add_parser("local")
    local.add_argument("iter", type=int, default=7000)
    client = subparsers.add_parser("client")
    client.add_argument("--ip", default="localhost")
    client.add_argument("--port", type=int, default=6009)
    server = subparsers.add_parser("server")
    server.add_argument("iter", type=int, default=7000)
    server.add_argument("--ip", default="localhost")
    server.add_argument("--port", type=int, default=6009)
    args = parser.parse_args()

    match args.mode:
        case "local":
            mode = ViewerMode.LOCAL
        case "client":
            mode = ViewerMode.CLIENT
        case "server":
            mode = ViewerMode.SERVER

    if mode is ViewerMode.CLIENT:
        viewer = GaussianViewer(mode)
    else:
        viewer = GaussianViewer.from_ply(lp.extract(args), pp.extract(args), args.iter, mode)

    viewer.run()
