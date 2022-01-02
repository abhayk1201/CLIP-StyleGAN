"""
Deep Image Prior Baseline
"""
from pathlib import Path

import clip
import hydra
import torch
import wandb
from omegaconf import DictConfig
from tqdm import tqdm

from loss import CLIPLoss
from loss import dump_metrics
from models import registry
from task_init import task_registry
from utils.catch_error import catch_error_decorator
from utils.data import load_latent_or_img, load_caption
from utils.train_helper import (
    train_setup,
    get_optimizer_lr_scheduler,
    setup_wandb,
    wandb_image,
    save_images,
)


@catch_error_decorator
@hydra.main(config_name=Path(__file__).stem, config_path="conf")
def main(cfg: DictConfig):
    device = train_setup(cfg)

    # Image (1CHW)
    img_gt = load_latent_or_img(**cfg.img)
    img_gt = img_gt.to(device)

    # CLIP model
    clip_loss = CLIPLoss(image_size=cfg.img.height, device=device)

    # Prior model (UNet, stylegan etc)
    prior_model = registry[cfg.model.name](**cfg.model.kwargs).to(device)

    # Preprocess
    caption = load_caption(cfg.img.caption)
    text = clip.tokenize([caption]).to(device)

    # Setup forward func
    img_gt, forward_func, metric = task_registry[cfg.task.name](
        img_gt, cfg.task, device
    )

    # wandb
    setup_wandb(cfg, img_gt, forward_func, caption)

    # Noise tensor
    noise_tensor = torch.rand(size=img_gt.shape).to(device)

    # Optimizer
    optim, lr_scheduler = get_optimizer_lr_scheduler(
        prior_model.parameters(), cfg.optim
    )

    # Train
    pbar = tqdm(total=cfg.train.num_steps)
    for step in range(cfg.train.num_steps):
        pbar.update(1)

        if cfg.optim.get("use_spherical", False):
            optim.opt.zero_grad()
        else:
            optim.zero_grad()

        img_out = prior_model(noise_tensor)

        # CLIP similarity
        loss_clip = clip_loss(img_out, text) * cfg.loss.clip

        # MSE
        loss_forward = metric(forward_func(img_gt), forward_func(img_out))
        loss_forward *= cfg.loss.forward

        loss = loss_forward + loss_clip
        loss.backward()

        optim.step()
        lr_scheduler.step()

        # pbar
        log_dict = {
            "loss_forward": loss_forward.item(),
            "loss_clip": loss_clip.item(),
            "loss": loss.item(),
        }
        pbar.set_description(" | ".join([f"{k}:{v:.3f}" for k, v in log_dict.items()]))

        if step % cfg.train.log_steps == 0:
            log_dict.update(
                {
                    "output": wandb_image(img_out, cfg.img.name),
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
