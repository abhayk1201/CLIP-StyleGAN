import hydra
from omegaconf import DictConfig

from stylegan_optimizer import load_stylegan
from task_init import task_registry
from utils.catch_error import catch_error_decorator
from utils.data import load_latent_or_img
from utils.train_helper import train_setup, save_images
from loguru import logger
from utils.timer import catchtime


@catch_error_decorator
@hydra.main(config_name="stylegan_optimizer", config_path="../conf")
def main(cfg: DictConfig):
    device = train_setup(cfg)

    # StyleGANv2
    g_ema = load_stylegan(cfg.stylegan).to(device)

    img_gt = load_latent_or_img(stylegan_gen=g_ema, **cfg.img)

    # Setup forward func
    img_gt, forward_func, metric = task_registry[cfg.task.name](img_gt, cfg.task, device)

    logger.info(f"Computing forward_func(groundtruth) for {cfg.task.name}")

    with catchtime() as t:
        img_gt_forward = forward_func(img_gt)

    logger.info(f"Elapsed time {t()} seconds.")
    logger.info("Saving groundtruth and forward_func(groundtruth)")

    # Dump gt, forward gt, out, forward out
    save_images(
        groundtruth=img_gt,
        groundtruth_forward=img_gt_forward,
    )


if __name__ == "__main__":
    main()
