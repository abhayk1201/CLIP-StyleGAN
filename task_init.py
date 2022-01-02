"""
Initialization functions for a Task

Eg: Drawing mask, getting downsampling function setup
"""

from functools import partial
from pathlib import Path

import numpy as np
from einops import rearrange
from loguru import logger
from matplotlib import pyplot as plt
from roipoly import RoiPoly
from torch.nn import functional as F
from kornia import filters
from utils.random_mask import generate_spiky_mask
import torch

from utils.bicubic import BicubicDownSample

task_registry = {}


def _register(*aliases):
    """
    Adds functions to task registry

    :param aliases: Other aliases besides the function name
    :return: register decorator
    """

    def decorator(func):
        task_registry[func.__name__] = func

        for name in aliases:
            task_registry[name] = func

        return func

    return decorator


@_register()
def super_resolution(img, task_cfg, device=torch.device("cpu")):
    forward_func = eval(task_cfg.get("downsample_func", BicubicDownSample))(
        task_cfg.factor
    )
    metric = eval(task_cfg.get("metric", F.mse_loss))

    return img, forward_func, metric


@_register("gaussian_deblurring")
def deblurring(img, task_cfg, device=torch.device("cpu")):
    forward_func = eval(
        task_cfg.get("blur_func", "filters.GaussianBlur2d((10, 10), 10)")
    )

    metric = eval(task_cfg.get("metric", F.mse_loss))

    return img, forward_func, metric


@_register()
def motion_deblurring(img, task_cfg, device=torch.device("cpu")):
    # Get filter
    kernel = filters.get_motion_kernel2d(
        task_cfg.kernel_size, task_cfg.angle, task_cfg.direction
    )
    # Pad and roll
    _, _, h, w = img.shape
    kernel = rearrange(kernel, "1 h w -> h w")

    # Correlation <---> convolution swap
    kernel = torch.flip(kernel, [0, 1])

    # Send to device
    kernel = kernel.to(device)

    def _simulate(image, kernel):
        assert image.ndim == 4, "Expected NCHW format"
        assert kernel.ndim == 2, "Expected HW format"

        _, _, h, w = image.shape

        kernel_h, kernel_w = kernel.shape

        image = F.pad(
            image, (kernel_w // 2, kernel_w // 2 + 1, kernel_h // 2, kernel_h // 2 + 1)
        )
        kernel = F.pad(kernel, (w // 2, w // 2, h // 2, h // 2))

        # Centre roll
        for dim in range(2):
            kernel = roll_n(kernel, axis=dim, n=kernel.size(dim) // 2)

        # Where mask is 1, nullify
        kernel = rearrange(kernel, "h w -> 1 1 h w")

        # Where mask is 1, nullify
        img_out = fft_conv2d(image, kernel)
        _, _, h_out, w_out = img_out.shape

        # Center crop
        img_out = img_out[
            :,
            :,
            h_out // 2 - h // 2 : h_out // 2 + h // 2,
            w_out // 2 - w // 2 : w_out // 2 + w // 2,
        ]

        return img_out

    metric = eval(task_cfg.get("metric", F.mse_loss))

    forward_func = partial(_simulate, kernel=kernel)

    return img, forward_func, metric


@_register()
def inpainting(img, task_cfg, device=torch.device("cpu")):
    # Rearrange, normalize to 0...1
    img_draw = rearrange(img.cpu().clone(), "1 c h w -> h w c")

    img_draw = torch.clip(img_draw, -1, 1)
    img_draw = (img_draw + 1) * 0.5
    # img_draw = (img_draw - img_draw.min()) / (img_draw.max() - img_draw.min())

    if not Path(task_cfg.mask.path).exists():
        logger.info(f"No mask found at {task_cfg.mask.path}")

        # Choose RoI
        if task_cfg.mask.generation == "manual":
            logger.info("Please draw mask")
            plt.imshow(img_draw)
            plt.title("Choose region to mask out")
            my_roi = RoiPoly(color="r")
            my_roi.display_roi()

            # Get mask
            mask = my_roi.get_mask(img_draw[:, :, 0])
        # Random mask
        elif task_cfg.mask.generation == "random":
            logger.info("Generating random mask")
            h, w, _ = img_draw.shape
            mask = generate_spiky_mask(h, w, **task_cfg.mask.kwargs)

        else:
            raise AssertionError

        np.save(task_cfg.mask.path, mask)
    else:
        logger.info(f"Loaded mask from {task_cfg.mask.path}")
        mask = np.load(task_cfg.mask.path)

    mask = torch.tensor(mask).to(device)
    # Forward func
    metric = eval(task_cfg.get("metric", "F.mse_loss"))

    def _mask_image(image, mask):
        assert image.ndim == 4, "Expected NCHW format"
        assert mask.ndim == 2, "Expected HW format"

        # Where mask is 1, nullify
        mask = rearrange(mask, "h w -> 1 1 h w")
        zeroed_image = torch.zeros_like(image)
        return torch.where(mask == 0, image, zeroed_image)

    forward_func = partial(_mask_image, mask=mask)

    return img, forward_func, metric


def fft_conv2d(input, kernel):
    """
    Source: https://github.com/siddiquesalman/flatnet/blob/d1c15ceeca00db57befa9db17bb02047a60a1a79/models/fftlayer.py#L19

    Computes the convolution in the frequency domain given
    Expects input and kernel already in frequency domain!
    :param input: shape (B, Cin, H, W)
    :param kernel: shape (Cout, Cin, H, W)
    :param bias: shape of (B, Cout, H, W)
    :return:
    """
    input = torch.fft.rfft2(input)
    kernel = torch.fft.rfft2(kernel)

    out = input * kernel

    out = torch.fft.irfft2(out)

    return out


def roll_n(X, axis, n):
    """
    Source: https://github.com/siddiquesalman/flatnet/blob/d1c15ceeca00db57befa9db17bb02047a60a1a79/utils/ops.py#L46
    :param X:
    :param axis:
    :param n:
    :return:
    """
    f_idx = tuple(
        slice(None, None, None) if i != axis else slice(0, n, None)
        for i in range(X.dim())
    )
    b_idx = tuple(
        slice(None, None, None) if i != axis else slice(n, None, None)
        for i in range(X.dim())
    )
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front], axis)


@_register()
def lensless(img, task_cfg, device=torch.device("cpu")):
    def _simulate(image, psf):
        assert image.ndim == 4, "Expected NCHW format"
        assert psf.ndim == 2, "Expected HW format"

        _, _, h, w = image.shape

        psf_h, psf_w = psf.shape

        image = F.pad(image, (psf_w // 2, psf_w // 2, psf_h // 2, psf_h // 2))
        psf = F.pad(psf, (w // 2, w // 2, h // 2, h // 2))

        # Centre roll
        for dim in range(2):
            psf = roll_n(psf, axis=dim, n=psf.size(dim) // 2)

        # Where mask is 1, nullify
        psf = rearrange(psf, "h w -> 1 1 h w")
        img_out = fft_conv2d(image, psf)
        _, _, h_out, w_out = img_out.shape

        # breakpoint()
        # # plot
        # img_plot = rearrange(img_out, "1 c h w -> h w c")
        # img_plot = (img_plot - img_plot.min()) / (img_plot.max() - img_plot.min())
        # plt.imshow(img_plot)
        # plt.show()

        # Center crop
        img_out = img_out[
            :,
            :,
            h_out // 2 - h // 2 : h_out // 2 + h // 2,
            w_out // 2 - w // 2 : w_out // 2 + w // 2,
        ]

        # Normalize
        img_out = (img_out - img_out.min()) / (img_out.max() - img_out.min())

        # 0...1 -> -1...1
        img_out = (img_out - 0.5) * 2

        return img_out

    # Load psf
    psf = np.load(task_cfg.get("psf_path"))
    psf = torch.tensor(psf).float().to(device)

    forward_func = partial(_simulate, psf=psf)

    metric = eval(task_cfg.get("metric", F.mse_loss))

    return img, forward_func, metric
