import os
from contextlib import contextmanager
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import wandb
from einops import rearrange
from omegaconf import OmegaConf
from torch.nn import functional as F
from torch.optim import Optimizer
import torchvision


@contextmanager
def _blank_context():
    yield


def get_device(device_str: str) -> torch.device:
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device(device_str)
    else:
        return torch.device("cpu")


def manual_seed(seed):
    # Manual seeds
    if seed is not None:
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True


class SphericalOptimizer(Optimizer):
    def __init__(self, optimizer, params, **kwargs):
        self.opt = optimizer(params, **kwargs)
        self.params = params
        with torch.no_grad():
            self.radii = {
                param: (
                    param.pow(2).sum(tuple(range(2, param.ndim)), keepdim=True) + 1e-9
                ).sqrt()
                for param in params
            }

    @torch.no_grad()
    def step(self, closure=None):
        loss = self.opt.step(closure)
        for param in self.params:
            param.data.div_(
                (
                    param.pow(2).sum(tuple(range(2, param.ndim)), keepdim=True) + 1e-9
                ).sqrt()
            )
            param.mul_(self.radii[param])

        return loss


def get_optimizer_lr_scheduler(
    params: List,
    optim_cfg: Dict,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    optim_dict = {
        "adam": torch.optim.Adam,
    }

    # Spherical Optimizer
    if optim_cfg.get("use_spherical"):
        optim = SphericalOptimizer(optim_dict[optim_cfg.name], params, lr=optim_cfg.lr)
    else:
        optim = optim_dict[optim_cfg.name](params, lr=optim_cfg.lr)

    # Source: https://github.com/marcin-laskowski/Pulse/blob/05eeab38c3a5e52055549c1b12b4f86427cf6883/PULSE.py#L127
    steps = optim_cfg.num_steps
    schedule_dict = {
        "fixed": lambda x: 1,
        "linear1cycle": lambda x: (9 * (1 - np.abs(x / steps - 1 / 2) * 2) + 1) / 10,
        "linear1cycledrop": lambda x: (
            9 * (1 - np.abs(x / (0.9 * steps) - 1 / 2) * 2) + 1
        )
        / 10
        if x < 0.9 * steps
        else 1 / 10 + (x - 0.9 * steps) / (0.1 * steps) * (1 / 1000 - 1 / 10),
    }
    schedule_func = schedule_dict[optim_cfg.schedule]

    ## TODO: hackish
    opt = optim.opt if optim_cfg.get("use_spherical") else optim
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(opt, schedule_func)

    return optim, lr_scheduler


def preprocess_for_CLIP(image):
    """
    pytorch-based preprocessing for CLIP

    Source: https://github.com/codestella/putting-nerf-on-a-diet/blob/90427f6fd828abb2f0b7966fc82753ff43edb338/nerf/clip_utils.py#L153
    Args:
        image [B, 3, H, W]: batch image
    Return
        image [B, 3, 224, 224]: pre-processed image for CLIP
    """
    device = image.device
    dtype = image.dtype

    mean = torch.tensor(
        [0.48145466, 0.4578275, 0.40821073], device=device, dtype=dtype
    ).reshape(1, 3, 1, 1)
    std = torch.tensor(
        [0.26862954, 0.26130258, 0.27577711], device=device, dtype=dtype
    ).reshape(1, 3, 1, 1)
    image = F.interpolate(
        image, (224, 224), mode="bicubic"
    )  # assume that images have rectangle shape.
    image = (image - mean) / std
    return image


def train_setup(cfg):
    if not cfg.get("silent"):
        print(OmegaConf.to_yaml(cfg))

    # Manual seeds
    manual_seed(cfg.get("seed"))

    # Set device
    device = get_device(cfg.device)

    # Log more frequently on CPU
    # Assuming you will be debugging then
    if device == torch.device("cpu"):
        cfg.train.log_steps = min(5, cfg.train.log_steps)

    return device


def wandb_image(img_tensor, caption: str = None):
    return [
        wandb.Image(
            img_tensor.detach(),
            caption=caption,
        )
    ]


def setup_wandb(cfg, img, forward_func, text):
    if cfg.wandb.use:
        with open(cfg.wandb.api_key) as f:
            os.environ["WANDB_API_KEY"] = f.read()

        wandb.init(
            entity=cfg.wandb.entity,
            config=OmegaConf.to_container(cfg, resolve=True),
            project=cfg.wandb.project,
            name=cfg.wandb.name,
            reinit=True,
            save_code=True,
        )

        caption_data = [[text]]
        columns = ["text input"]

        table = wandb.Table(data=caption_data, columns=columns)

        wandb.log(
            {
                "groundtruth": wandb_image(img, cfg.img.name),
                "input_image": wandb_image(forward_func(img), cfg.img.name),
                "caption_table": table,
            },
            step=0,
        )


def save_images(**kwargs):
    for name, tensor in kwargs.items():
        torchvision.utils.save_image(tensor, f"{name}.png", normalize=True, range=(-1, 1))
