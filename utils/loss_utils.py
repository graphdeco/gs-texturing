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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from gaussian_renderer import render
from utils.image_utils import psnr
from scene.gaussian_model import GaussianModel
from typing import List, Tuple
from scene.cameras import Camera


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True, aggregate=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    ssim_map = _ssim(img1, img2, window, window_size, channel)

    if aggregate == False:
        return ssim_map

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def _ssim(img1, img2, window, window_size, channel):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    return ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))


def evaluate_model(model: GaussianModel, cameras: list[Camera], pipe, background: torch.Tensor) -> tuple[float, float, float]:
    if cameras and len(cameras) > 0:
        l1_test = torch.tensor([0.0], device="cuda")
        ssim_test = torch.tensor([0.0], device="cuda")
        psnr_test = torch.tensor([0.0], device="cuda")

        for viewpoint in cameras:
            l1, ssim, psnr, _, _ = evaluate_viewpoint(model, viewpoint, pipe, background)
            l1_test += l1
            ssim_test += ssim
            psnr_test += psnr

        l1_test /= len(cameras)
        ssim_test /= len(cameras)
        psnr_test /= len(cameras)
    return l1_test.item(), ssim_test.item(), psnr_test.item()

# Please somehow remove pipe background
def evaluate_viewpoint(model: GaussianModel, viewpoint: Camera, pipe, background):
    image = torch.clamp(render(viewpoint, model, pipe, background)["render"], 0.0, 1.0)
    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
    l1_score = l1_loss(image, gt_image).mean().double()
    ssim_score = (1.0 - ssim(image, gt_image)).mean().double()
    psnr_score = psnr(image, gt_image).mean().double()
    return l1_score, ssim_score, psnr_score, image, gt_image
