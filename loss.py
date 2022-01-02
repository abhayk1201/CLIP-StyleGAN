import json

import clip
import kornia
import lpips
import torch
import pytorch_msssim
from loguru import logger


class CLIPLoss(torch.nn.Module):
    def __init__(self, image_size, device=torch.device("cpu")):
        super(CLIPLoss, self).__init__()
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.upsample = torch.nn.Upsample(scale_factor=7)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=image_size // 32)

    def forward(self, image, text):
        image = self.avg_pool(self.upsample(image))
        similarity = 1 - self.model(image, text)[0] / 100
        return similarity


def LossGeocross(latent):
    """
    Uses geodesic distance on sphere to sum pairwise distances of the 18 vectors
    """
    if latent.shape[1] == 1:
        return 0
    else:
        _, n_latent, latent_dim = latent.shape
        X = latent.view(-1, 1, n_latent, latent_dim)
        Y = latent.view(-1, n_latent, 1, latent_dim)
        A = ((X - Y).pow(2).sum(-1) + 1e-9).sqrt()
        B = ((X + Y).pow(2).sum(-1) + 1e-9).sqrt()
        D = 2 * torch.atan2(A, B)
        D = ((D.pow(2) * latent_dim).mean((1, 2)) / 8.0).sum()
        return D


def dump_metrics(img_out, img_gt, text, clip_loss, device=torch.device("cpu")):
    """
    Dumps all metrics

    :param img_out: Output image NCHW
    :param img_gt: GT image NCHW
    :param text: Caption supplied
    :param device: Accelerator
    """
    metrics_dict = {}

    # -1...1 -> 0...1
    img_gt_normalized = (img_gt + 1) * 0.5
    img_gt_normalized = torch.clamp(img_gt_normalized, min=0, max=1)
    img_out_normalized = (img_out + 1) * 0.5
    img_out_normalized = torch.clamp(img_out_normalized, min=0, max=1)

    # LPIPS
    loss_fn_alex = lpips.LPIPS(net="alex").to(device)
    metrics_dict["lpips_score"] = loss_fn_alex(img_out, img_gt).item()

    # PSNR
    metrics_dict["psnr_score"] = kornia.metrics.psnr(
        img_out_normalized, img_gt_normalized, max_val=1
    ).item()

    # SSIM
    # Recommended window size 11
    # from Wang (2004), "Image quality assessment: from error visibility to structural similarity"
    metrics_dict["ssim_score"] = (
        kornia.metrics.ssim(img_out_normalized, img_gt_normalized, window_size=11)
        .mean()
        .item()
    )

    # CLIP similarity
    metrics_dict["clip_score"] = clip_loss(img_out, text).item()

    # Pytorch MS-SSIM
    # https://github.com/jorge-pessoa/pytorch-msssim
    metrics_dict["ms_ssim"] = pytorch_msssim.msssim(
        img_out_normalized, img_gt_normalized
    ).item()

    with open("metrics.json", "w") as f:
        json.dump(metrics_dict, f, sort_keys=True, indent=4)

    logger.info(" | ".join([f"{k}:{v:.3f}" for k, v in metrics_dict.items()]))

    return metrics_dict
