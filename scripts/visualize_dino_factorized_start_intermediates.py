#!/usr/bin/env python3
"""Visualize DINO factorized sphere start intermediates from a JiT checkpoint."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from main_jit_vae_start import (
    HfParquetImageNetDataset,
    ImageListWithLabels,
    _tokens_to_norm_heatmap,
    _tokens_to_pca_rgb,
    build_parser,
    build_vae_start_encoder,
    predict_x0,
    sample_vae_start,
    standard_normal_moment_kl,
    standard_normal_sigreg_loss,
)
from utils.builders import create_generation_model
from utils.data_util import center_crop_arr
from utils.start_util import apply_start_support, sample_start


def _loader(args):
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
            raise FileNotFoundError(f"no ImageNet train data under {data_path}")
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


def _signed_heatmap(x: torch.Tensor, image_size: int = 256) -> torch.Tensor:
    x = x.detach().float()
    max_abs = x.flatten(1).abs().amax(dim=1).view(-1, 1, 1, 1).clamp_min(1e-6)
    value = (x / max_abs).clamp(-1, 1)
    red = value.clamp_min(0)
    blue = (-value).clamp_min(0)
    green = 1.0 - value.abs()
    rgb = torch.cat([red, green.clamp(0, 1), blue], dim=1)
    return F.interpolate(rgb, size=(image_size, image_size), mode="nearest")


def _vector_to_grid(v: torch.Tensor) -> torch.Tensor:
    side = int(v.shape[1] ** 0.5)
    if side * side == v.shape[1]:
        return v.reshape(v.shape[0], 1, side, side)
    height = 16
    width = (v.shape[1] + height - 1) // height
    padded = F.pad(v, (0, height * width - v.shape[1]))
    return padded.reshape(v.shape[0], 1, height, width)


def _local_norm_heatmap(local: torch.Tensor, coarse_grid: int) -> torch.Tensor:
    norm = local.detach().float().norm(dim=-1).reshape(local.shape[0], 1, coarse_grid, coarse_grid)
    return _signed_heatmap(norm - norm.mean(dim=(2, 3), keepdim=True))


def _compact_rows(label: str, encoder, compact: torch.Tensor, nrow: int) -> list[Image.Image]:
    rows: list[Image.Image] = []
    if hasattr(encoder, "_split_compact"):
        global_code, local = encoder._split_compact(compact.detach())
        rows.append(_row(f"{label} compact global signed", _signed_heatmap(_vector_to_grid(global_code)), nrow))
        if local is not None:
            rows.append(_row(f"{label} compact local PCA", _tokens_to_pca_rgb(local), nrow))
            rows.append(_row(f"{label} compact local norm", _local_norm_heatmap(local, encoder.coarse_grid), nrow))
    else:
        rows.append(_row(f"{label} compact signed", _signed_heatmap(_vector_to_grid(compact.detach())), nrow))
    return rows


def _row(label: str, images: torch.Tensor, nrow: int, label_width: int = 330) -> Image.Image:
    grid = make_grid(images.detach().cpu(), nrow=nrow, padding=2, pad_value=1.0)
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
    path.parent.mkdir(parents=True, exist_ok=True)
    width = max(row.width for row in rows)
    height = sum(row.height for row in rows)
    canvas = Image.new("RGB", (width, height), "white")
    y = 0
    for row in rows:
        canvas.paste(row, (0, y))
        y += row.height
    canvas.save(path)


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.detach().float().flatten(1), b.detach().float().flatten(1), dim=1).mean().item())


def main():
    parser = argparse.ArgumentParser(
        "Visualize DINO factorized start intermediates",
        parents=[build_parser()],
        conflict_handler="resolve",
    )
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--num_images", default=8, type=int)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--seed", default=123, type=int)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.global_bsz = args.num_images
    args.batch_size = args.num_images
    args.enable_wandb = False
    args.disable_vis = True
    args.output_dir = str(args.out_dir)
    args.project = "dino_start_intermediate_vis"
    args.exp_name = "vis"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    config = ckpt.get("vae_start_config", {})
    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, value)
    if args.vae_start_encoder_type != "dino_factorized_patch_sphere":
        raise ValueError(f"expected dino_factorized_patch_sphere, got {args.vae_start_encoder_type}")

    model, _ = create_generation_model(args)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval().requires_grad_(False)

    encoder = build_vae_start_encoder(args).cuda()
    encoder.load_state_dict(ckpt["vae_start_encoder"], strict=True)
    encoder.eval().requires_grad_(False)

    x0, labels = next(iter(_loader(args)))
    x0 = x0.cuda(non_blocking=True)[: args.num_images]
    labels = labels.cuda(non_blocking=True)[: args.num_images]
    ones = torch.ones(x0.shape[0], device=x0.device)
    random_labels = torch.randint(0, args.num_classes, labels.shape, device=labels.device)

    with torch.no_grad():
        noisy_raw, clean_raw, _ = sample_vae_start(
            encoder, x0, args.vae_start_sample_mode, args.vae_start_mean_scale,
        )
        aux = encoder._last_start_sample
        noisy_supported = apply_start_support(noisy_raw, mode=args.start_support_mode, noise_scale=args.noise_scale)
        clean_supported = apply_start_support(clean_raw, mode=args.start_support_mode, noise_scale=args.noise_scale)
        random_bridge_raw = aux["random_start"]
        random_bridge_supported = apply_start_support(
            random_bridge_raw, mode=args.start_support_mode, noise_scale=args.noise_scale,
        )
        random_pixel = sample_start(
            tuple(x0.shape), device=x0.device, dtype=x0.dtype,
            noise_scale=args.noise_scale, mode=args.start_support_mode,
        )

        recon_noisy = predict_x0(model, noisy_supported, ones, labels, drop_labels=False)
        recon_clean = predict_x0(model, clean_supported, ones, labels, drop_labels=False)
        out_random_bridge = predict_x0(model, random_bridge_supported, ones, random_labels, drop_labels=False)
        out_random_pixel = predict_x0(model, random_pixel, ones, random_labels, drop_labels=False)

        stats = {
            "checkpoint": str(args.ckpt),
            "num_images": int(x0.shape[0]),
            "start_support_mode": args.start_support_mode,
            "compact_dim": int(aux["compact_clean"].shape[1]),
            "compact_clean_std": float(aux["compact_clean"].std().item()),
            "compact_noisy_std": float(aux["compact_noisy"].std().item()),
            "compact_random_std": float(aux["compact_random"].std().item()),
            "compact_clean_noisy_cosine": _cos(aux["compact_clean"], aux["compact_noisy"]),
            "compact_clean_random_cosine": _cos(aux["compact_clean"], aux["compact_random"]),
            "clean_raw_std": float(clean_raw.std().item()),
            "noisy_raw_std": float(noisy_raw.std().item()),
            "random_bridge_raw_std": float(random_bridge_raw.std().item()),
            "clean_supported_std": float(clean_supported.std().item()),
            "noisy_supported_std": float(noisy_supported.std().item()),
            "random_bridge_supported_std": float(random_bridge_supported.std().item()),
            "random_pixel_std": float(random_pixel.std().item()),
            "sigreg_noisy_supported": float(standard_normal_sigreg_loss(noisy_supported, patch_size=args.vae_start_sigreg_patch_size or args.patch_size, num_slices=args.vae_start_sigreg_num_slices, max_samples=args.vae_start_sigreg_max_samples).item()),
            "sigreg_clean_supported": float(standard_normal_sigreg_loss(clean_supported, patch_size=args.vae_start_sigreg_patch_size or args.patch_size, num_slices=args.vae_start_sigreg_num_slices, max_samples=args.vae_start_sigreg_max_samples).item()),
            "sigreg_random_bridge_supported": float(standard_normal_sigreg_loss(random_bridge_supported, patch_size=args.vae_start_sigreg_patch_size or args.patch_size, num_slices=args.vae_start_sigreg_num_slices, max_samples=args.vae_start_sigreg_max_samples).item()),
            "sigreg_random_pixel": float(standard_normal_sigreg_loss(random_pixel, patch_size=args.vae_start_sigreg_patch_size or args.patch_size, num_slices=args.vae_start_sigreg_num_slices, max_samples=args.vae_start_sigreg_max_samples).item()),
            "input_kl_noisy_supported": float(standard_normal_moment_kl(noisy_supported, args.vae_start_input_kl_granularity).item()),
            "cycle_mse_noisy": float(torch.mean((recon_noisy - x0) ** 2).item()),
            "cycle_mse_clean": float(torch.mean((recon_clean - x0) ** 2).item()),
            "random_bridge_out_std": float(out_random_bridge.std().item()),
            "random_pixel_out_std": float(out_random_pixel.std().item()),
        }

    nrow = min(4, args.num_images)
    rows: list[Image.Image] = [_row("real x0", _to_image_range(x0), nrow)]
    rows += _compact_rows("clean", encoder, aux["compact_clean"], nrow)
    rows += _compact_rows("noisy", encoder, aux["compact_noisy"], nrow)
    rows += _compact_rows("random", encoder, aux["compact_random"], nrow)
    rows.extend([
        _row("expanded clean tokens PCA", _tokens_to_pca_rgb(aux["latent_clean"]), nrow),
        _row("expanded clean tokens norm", _tokens_to_norm_heatmap(aux["latent_clean"]), nrow),
        _row("expanded noisy tokens PCA", _tokens_to_pca_rgb(aux["latent_noisy"]), nrow),
        _row("expanded noisy tokens norm", _tokens_to_norm_heatmap(aux["latent_noisy"]), nrow),
        _row("expanded random tokens PCA", _tokens_to_pca_rgb(aux["latent_random"]), nrow),
        _row("expanded random tokens norm", _tokens_to_norm_heatmap(aux["latent_random"]), nrow),
        _row("clean bridge raw B(z_clean)", _to_noise_range(clean_raw), nrow),
        _row("clean bridge supported", _to_noise_range(clean_supported), nrow),
        _row("noisy bridge raw B(z_noisy)", _to_noise_range(noisy_raw), nrow),
        _row("noisy bridge supported", _to_noise_range(noisy_supported), nrow),
        _row("random bridge raw B(z_rand)", _to_noise_range(random_bridge_raw), nrow),
        _row("random bridge supported", _to_noise_range(random_bridge_supported), nrow),
        _row("random pixel supported", _to_noise_range(random_pixel), nrow),
        _row("JiT(noisy bridge, paired label)", _to_image_range(recon_noisy), nrow),
        _row("JiT(clean bridge, paired label)", _to_image_range(recon_clean), nrow),
        _row("JiT(random bridge, random label)", _to_image_range(out_random_bridge), nrow),
        _row("JiT(random pixel, random label)", _to_image_range(out_random_pixel), nrow),
    ])
    _save_contact(rows, args.out_dir / "dino_factorized_intermediate_contact.png")
    (args.out_dir / "dino_factorized_intermediate_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"saved {args.out_dir / 'dino_factorized_intermediate_contact.png'}")


if __name__ == "__main__":
    main()
