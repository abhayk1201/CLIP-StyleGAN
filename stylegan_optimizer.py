"""
Proposed Method
"""
from pathlib import Path

import clip
import hydra
import torch
import wandb
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm

from loss import CLIPLoss, LossGeocross, dump_metrics
from models.stylegan import Generator
from task_init import task_registry
from utils.catch_error import catch_error_decorator
from utils.data import load_latent_or_img, load_caption
from utils.train_helper import (
    get_optimizer_lr_scheduler,
    train_setup,
    wandb_image,
    setup_wandb,
    save_images,
)


def set_noise_vars(g_ema, stylegan_cfg):
    """
    Setup trainable and non-trainable noise tensors for stylegan

    :param g_ema: stylegan generator
    :param stylegan_cfg: stylegan config
    :return:
        noise_ll: Set of all noise tensors
        noise_var_ll: Set of trainable (requires_grad=True) noise tensors
    """
    # Trainable noise (adds to style)
    num_layers = g_ema.num_layers
    num_trainable_noise_layers = stylegan_cfg.num_trainable_noise_layers
    noise_type = stylegan_cfg.noise_type
    bad_noise_layers = stylegan_cfg.bad_noise_layers

    noise_ll = []
    noise_var_ll = []

    for i in range(num_layers):
        noise_tensor = getattr(g_ema.noises, f"noise_{i}")

        if (noise_type == "zero") or (i in bad_noise_layers):
            new_noise = torch.zeros_like(noise_tensor)
            new_noise.requires_grad = False
        elif noise_type == "fixed":
            new_noise = noise_tensor
        elif noise_type == "trainable":
            new_noise = noise_tensor.detach().clone()
            if i < num_trainable_noise_layers:
                new_noise.requires_grad = True
                noise_var_ll.append(new_noise)
            else:
                new_noise.requires_grad = False
        else:
            raise Exception(f"unknown noise type {noise_type}")

        noise_ll.append(new_noise)

    return noise_ll, noise_var_ll


def load_stylegan(stylegan_cfg):
    g_ema = Generator(stylegan_cfg.size, 512, 8)
    logger.info(f"Loading StyleGANv2 ckpt from {stylegan_cfg.ckpt}")
    stylegan_ckpt = torch.load(stylegan_cfg.ckpt, map_location="cpu")["g_ema"]
    g_ema.load_state_dict(stylegan_ckpt, strict=False)
    g_ema.eval()

    return g_ema


@catch_error_decorator
@hydra.main(config_name=Path(__file__).stem, config_path="conf")
def main(cfg: DictConfig):
    device = train_setup(cfg)

    # StyleGANv2
    g_ema = load_stylegan(cfg.stylegan).to(device)

    img_gt = load_latent_or_img(stylegan_gen=g_ema, device=device, **cfg.img)

    # Random init, from where optimization begins
    random_latent = g_ema.random_latent().clone().detach()
    random_latent = random_latent.unsqueeze(0).repeat(1, g_ema.n_latent, 1)
    random_latent.requires_grad = True

    # Setup forward func
    img_gt, forward_func, metric = task_registry[cfg.task.name](
        img_gt, cfg.task, device
    )
    img_gt = img_gt.to(device)

    # CLIP model
    clip_loss = CLIPLoss(image_size=cfg.stylegan.size, device=device)

    # Preprocess
    caption = load_caption(cfg.img.caption)
    text = clip.tokenize([caption]).to(device)

    # wandb
    setup_wandb(cfg, img_gt, forward_func, caption)

    # Noise tensors, including those that require gradients
    noise_ll, noise_var_ll = set_noise_vars(g_ema, cfg.stylegan)
    var_list = [random_latent] + noise_var_ll

    # Optimizer
    optim, lr_scheduler = get_optimizer_lr_scheduler(var_list, cfg.optim)

    # Train
    pbar = tqdm(total=cfg.train.num_steps)
    for step in range(cfg.train.num_steps):
        pbar.update(1)

        if cfg.optim.get("use_spherical", False):
            optim.opt.zero_grad()
        else:
            optim.zero_grad()

        img_out, _ = g_ema(
            [random_latent],
            input_is_latent=True,
            noise_tensor_ll=noise_ll,
            randomize_noise=False,
        )

        # CLIP similarity
        loss_clip = clip_loss(img_out, text) * cfg.loss.clip

        # MSE
        loss_forward = metric(forward_func(img_gt), forward_func(img_out))
        loss_forward *= cfg.loss.forward

        # Geocross
        loss_geocross = LossGeocross(random_latent) * cfg.loss.geocross

        loss = loss_forward + loss_clip + loss_geocross
        loss.backward()

        optim.step()
        lr_scheduler.step()

        # pbar
        log_dict = {
            "loss_forward": loss_forward.item(),
            "loss_clip": loss_clip.item(),
            "loss_geocross": loss_geocross.item(),
            "loss": loss.item(),
        }
        pbar.set_description(" | ".join([f"{k}:{v:.3f}" for k, v in log_dict.items()]))

        if step % cfg.train.log_steps == 0:
            log_dict.update(
                {
                    "output": wandb_image(torch.clip(img_out, -1, 1), cfg.img.name),
                    "output_forward": wandb_image(forward_func(img_out), cfg.img.name),
                }
            )
            wandb.log(log_dict, step=step)

    # Collate metrics
    metrics_dict = dump_metrics(img_out, img_gt, text, clip_loss, device)
    if cfg.wandb.use:
        wandb.run.summary.update(metrics_dict)

    # Dump gt, forward gt, out, forward out
    save_images(
        groundtruth=img_gt,
        groundtruth_forward=forward_func(img_gt),
        recovered=img_out,
        recovered_forward=forward_func(img_out),
    )


if __name__ == "__main__":
    main()
