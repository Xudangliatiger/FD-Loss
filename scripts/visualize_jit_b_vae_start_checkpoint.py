#!/usr/bin/env python3
"""Visualize the pixel-space VAE-start encoder saved in a JiT-B checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from main_jit_vae_start import (
    HfParquetImageNetDataset,
    ImageListWithLabels,
    VariationalStartEncoder,
    build_parser,
    gaussian_kl_per_dim,
    predict_x0,
    sample_vae_start,
    shuffle_vae_stats,
)
from utils.builders import create_generation_model
from utils.checkpoint_util import ckpt_resume


def _imagefolder_or_parquet_loader(args):
    import glob
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms
    from utils.data_util import center_crop_arr

    transform = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, args.img_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2.0 - 1.0),
    ])
    data_path = Path(args.data_path)
    if (data_path / "train").is_dir():
        dataset = datasets.ImageFolder(data_path / "train", transform=transform)
    elif args.image_list and args.image_root:
        dataset = ImageListWithLabels(args.image_list, args.image_root, transform=transform)
    else:
        parquet_files = sorted(glob.glob(str(data_path / "data" / "train-*.parquet")))
        if not parquet_files:
            raise FileNotFoundError(
                f"Could not find ImageFolder train/ or HF parquet train shards under {data_path}"
            )
        dataset = HfParquetImageNetDataset(parquet_files, args.img_size)
    return DataLoader(
        dataset,
        batch_size=args.num_images,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )


def _to_image_range(x: torch.Tensor) -> torch.Tensor:
    return (x.detach().float().clamp(-1, 1) + 1.0) * 0.5


def _to_noise_range(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().float()
    flat = x.flatten(1)
    lo = flat.min(dim=1).values.view(-1, 1, 1, 1)
    hi = flat.max(dim=1).values.view(-1, 1, 1, 1)
    return ((x - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1)


def _row(label: str, images: torch.Tensor, nrow: int, label_width: int = 260) -> Image.Image:
    grid = make_grid(images.cpu(), nrow=nrow, padding=2, pad_value=1.0)
    grid = (grid.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype("uint8")
    grid_img = Image.fromarray(grid)
    canvas = Image.new("RGB", (label_width + grid_img.width, grid_img.height), "white")
    canvas.paste(grid_img, (label_width, 0))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    draw.multiline_text((12, 12), label, fill=(20, 20, 20), font=font, spacing=4)
    return canvas


def _save_contact(rows: list[Image.Image], path: Path):
    width = max(r.width for r in rows)
    height = sum(r.height for r in rows)
    canvas = Image.new("RGB", (width, height), "white")
    y = 0
    for row in rows:
        canvas.paste(row, (0, y))
        y += row.height
    canvas.save(path)


def _cosine_to_x0(x: torch.Tensor, x0: torch.Tensor) -> float:
    return float(
        torch.nn.functional.cosine_similarity(x.flatten(1), x0.flatten(1), dim=1)
        .mean()
        .item()
    )


def main():
    parser = argparse.ArgumentParser(
        "Visualize JiT-B VAE-start checkpoint",
        parents=[build_parser()],
        conflict_handler="resolve",
    )
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--num_images", default=16, type=int)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--seed", default=123, type=int)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    args.global_bsz = args.num_images
    args.batch_size = args.num_images
    args.disable_vis = True
    args.enable_wandb = False
    args.output_dir = str(args.out_dir)
    args.project = "vae_start_vis"
    args.exp_name = "vis"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if "vae_start_encoder" not in ckpt:
        raise KeyError(f"{args.ckpt} does not contain 'vae_start_encoder'")

    if "model" not in ckpt and "base_model_ckpt" in ckpt and not args.load_from:
        args.load_from = ckpt["base_model_ckpt"]

    model, ema_model = create_generation_model(args)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=False)
    else:
        ckpt_resume(args, model, optimizer=None, model_ema=ema_model)
    model.eval().requires_grad_(False)

    config = ckpt.get("vae_start_config", {})
    encoder = VariationalStartEncoder(
        channels=3,
        hidden=int(config.get("vae_start_hidden", args.vae_start_hidden)),
        logvar_min=float(config.get("vae_start_logvar_min", args.vae_start_logvar_min)),
        logvar_max=float(config.get("vae_start_logvar_max", args.vae_start_logvar_max)),
    ).cuda()
    encoder.load_state_dict(ckpt["vae_start_encoder"])
    encoder.eval().requires_grad_(False)

    loader = _imagefolder_or_parquet_loader(args)
    x0, labels = next(iter(loader))
    x0 = x0.cuda(non_blocking=True)[: args.num_images]
    labels = labels.cuda(non_blocking=True)[: args.num_images]
    ones = torch.ones(x0.shape[0], device=x0.device)

    with torch.no_grad():
        z_start, mu, logvar = sample_vae_start(
            encoder,
            x0,
            mode=args.vae_start_sample_mode,
            mean_scale=args.vae_start_mean_scale,
        )
        sigma = torch.exp(0.5 * logvar)
        mu_shuffled, logvar_shuffled = shuffle_vae_stats(mu, logvar)
        sigma_shuffled = torch.exp(0.5 * logvar_shuffled)
        z_shuffled = mu_shuffled + sigma_shuffled * torch.randn_like(mu_shuffled)
        eps = torch.randn_like(x0)
        random_recon = predict_x0(model, eps, ones, labels, drop_labels=False)
        vae_recon = predict_x0(model, z_start, ones, labels, drop_labels=False)
        shuffled_recon = predict_x0(model, z_shuffled, ones, labels, drop_labels=False)
        mu_recon = predict_x0(model, mu, ones, labels, drop_labels=False)
        injected = z_start - eps

        stats = {
            "checkpoint": str(args.ckpt),
            "num_images": int(x0.shape[0]),
            "cycle_mse_z_start": float(torch.mean((vae_recon - x0) ** 2).item()),
            "cycle_mse_z_shuffled": float(torch.mean((shuffled_recon - x0) ** 2).item()),
            "cycle_mse_mu": float(torch.mean((mu_recon - x0) ** 2).item()),
            "random_start_mse": float(torch.mean((random_recon - x0) ** 2).item()),
            "kl_per_dim": float(gaussian_kl_per_dim(mu, logvar).item()),
            "mu_mean": float(mu.mean().item()),
            "mu_std": float(mu.std().item()),
            "logvar_mean": float(logvar.mean().item()),
            "logvar_std": float(logvar.std().item()),
            "sigma_mean": float(sigma.mean().item()),
            "sigma_std": float(sigma.std().item()),
            "z_start_mean": float(z_start.mean().item()),
            "z_start_std": float(z_start.std().item()),
            "z_shuffled_mean": float(z_shuffled.mean().item()),
            "z_shuffled_std": float(z_shuffled.std().item()),
            "mu_x0_cosine": _cosine_to_x0(mu, x0),
            "mu_shuffled_x0_cosine": _cosine_to_x0(mu_shuffled, x0),
            "z_start_x0_cosine": _cosine_to_x0(z_start, x0),
            "z_shuffled_x0_cosine": _cosine_to_x0(z_shuffled, x0),
            "injected_mean": float(injected.mean().item()),
            "injected_std": float(injected.std().item()),
        }

    nrow = min(8, args.num_images)
    rows = [
        _row("real x0", _to_image_range(x0), nrow),
        _row("VAE mean mu(x0)", _to_noise_range(mu), nrow),
        _row("VAE sigma(x0)", _to_noise_range(sigma), nrow),
        _row("VAE logvar(x0)", _to_noise_range(logvar), nrow),
        _row("sampled z_start", _to_noise_range(z_start), nrow),
        _row("shuffled z_start", _to_noise_range(z_shuffled), nrow),
        _row("injected z_start - eps", _to_noise_range(injected), nrow),
        _row("JiT(z_start, t=1)", _to_image_range(vae_recon), nrow),
        _row("JiT(shuffled z_start, t=1)", _to_image_range(shuffled_recon), nrow),
        _row("JiT(mu, t=1)", _to_image_range(mu_recon), nrow),
        _row("JiT(random eps, t=1)", _to_image_range(random_recon), nrow),
    ]
    _save_contact(rows, args.out_dir / "vae_start_effect_contact.png")
    (args.out_dir / "vae_start_effect_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"saved {args.out_dir / 'vae_start_effect_contact.png'}")


if __name__ == "__main__":
    main()
