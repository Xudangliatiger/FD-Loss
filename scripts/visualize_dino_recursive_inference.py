#!/usr/bin/env python3
"""Visualize recursive DINO-sphere inference from a JiT checkpoint.

The probe starts from a random sphere bridge start, generates one image with
JiT, feeds that image back through the frozen DINO-sphere encoder/bridge, and
generates again. It is intended to diagnose whether the DINO frontend acts as a
self-refinement map or simply amplifies artifacts.
"""

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
from torchvision.utils import make_grid

from main_jit_vae_start import (
    _fixed_inference_labels,
    build_parser,
    build_vae_start_encoder,
)
from utils.builders import create_generation_model
from utils.checkpoint_util import ckpt_resume
from utils.start_util import apply_start_support

_DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def _to_image_range(x: torch.Tensor) -> torch.Tensor:
    return (x.detach().float().clamp(-1, 1) + 1.0) * 0.5


def _to_noise_range(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().float()
    flat = x.flatten(1)
    lo = flat.min(dim=1).values.view(-1, 1, 1, 1)
    hi = flat.max(dim=1).values.view(-1, 1, 1, 1)
    return ((x - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1)


def _row(label: str, images: torch.Tensor, nrow: int, label_width: int = 310) -> Image.Image:
    grid = make_grid(images.cpu(), nrow=nrow, padding=2, pad_value=1.0)
    grid = (grid.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype("uint8")
    grid_img = Image.fromarray(grid)
    canvas = Image.new("RGB", (label_width + grid_img.width, grid_img.height), "white")
    canvas.paste(grid_img, (label_width, 0))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 17)
    except OSError:
        font = ImageFont.load_default()
    draw.multiline_text((12, 12), label, fill=(20, 20, 20), font=font, spacing=4)
    return canvas


def _save_contact(rows: list[Image.Image], path: Path):
    width = max(row.width for row in rows)
    height = sum(row.height for row in rows)
    canvas = Image.new("RGB", (width, height), "white")
    y = 0
    for row in rows:
        canvas.paste(row, (0, y))
        y += row.height
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _load_checkpointed_model(args, ckpt: dict) -> torch.nn.Module:
    if "model" not in ckpt and "base_model_ckpt" in ckpt and not args.load_from:
        args.load_from = ckpt["base_model_ckpt"]
    model, ema_model = create_generation_model(args)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=False)
    else:
        ckpt_resume(args, model, optimizer=None, model_ema=ema_model)
    model.eval().requires_grad_(False)
    return model


def _load_checkpointed_encoder(args, ckpt: dict) -> torch.nn.Module:
    if "vae_start_encoder" not in ckpt:
        raise KeyError("checkpoint does not contain 'vae_start_encoder'")
    config = ckpt.get("vae_start_config", {})
    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, value)
    encoder = build_vae_start_encoder(args).cuda()
    encoder.load_state_dict(ckpt["vae_start_encoder"])
    encoder.eval().requires_grad_(False)
    if getattr(encoder, "start_kind", None) != "sphere_latent":
        raise ValueError(f"expected sphere_latent encoder, got {getattr(encoder, 'start_kind', None)}")
    if not hasattr(encoder, "random_start"):
        raise ValueError("encoder does not implement random_start")
    return encoder


def _support(args, start: torch.Tensor) -> torch.Tensor:
    return apply_start_support(
        start,
        mode=args.start_support_mode,
        noise_scale=args.noise_scale,
    )


def _sample_reencoded_start(args, encoder: torch.nn.Module, image: torch.Tensor, mode: str) -> torch.Tensor:
    sample = encoder.sample_start(image.clamp(-1, 1))
    if mode == "clean":
        return _support(args, sample["clean_start"])
    if mode == "noisy":
        return _support(args, sample["start"])
    if mode == "random":
        return _support(args, sample["random_start"])
    raise ValueError(f"unknown reencode mode: {mode}")


def _mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.mean((a.detach().float() - b.detach().float()) ** 2).item())


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(
        torch.nn.functional.cosine_similarity(
            a.detach().float().flatten(1),
            b.detach().float().flatten(1),
            dim=1,
        ).mean().item()
    )


def main():
    parser = argparse.ArgumentParser(
        "Visualize recursive DINO-sphere JiT inference",
        parents=[build_parser()],
        conflict_handler="resolve",
    )
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--num_images", default=16, type=int)
    parser.add_argument("--num_rounds", default=3, type=int)
    parser.add_argument(
        "--reencode_mode",
        choices=["clean", "noisy", "random"],
        default="clean",
        help="which DINO bridge start to use after re-encoding each generated image",
    )
    parser.add_argument("--seed", default=123, type=int)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.global_bsz = args.num_images
    args.batch_size = args.num_images
    args.disable_vis = True
    args.enable_wandb = False
    args.enable_amp = args.dtype != "fp32"
    args.amp_dtype = _DTYPE_MAP[args.dtype]
    args.output_dir = str(args.out_dir)
    args.project = "dino_recursive_inference"
    args.exp_name = "vis"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = _load_checkpointed_model(args, ckpt)
    encoder = _load_checkpointed_encoder(args, ckpt)

    device = torch.device("cuda")
    n = args.num_images
    labels = _fixed_inference_labels(args, n, device)
    nrow = min(8, n)

    rows: list[Image.Image] = []
    stats: dict[str, object] = {
        "checkpoint": str(args.ckpt),
        "num_images": int(n),
        "num_rounds": int(args.num_rounds),
        "reencode_mode": args.reencode_mode,
        "num_sampling_steps": int(args.num_sampling_steps),
        "cfg": float(args.cfg),
        "start_support_mode": args.start_support_mode,
        "labels": [int(x) for x in labels.detach().cpu().tolist()],
        "rounds": [],
    }

    with torch.inference_mode():
        start = encoder.random_start(n, device, torch.float32)
        start = _support(args, start)
        rows.append(_row("round 0\nrandom sphere bridge start", _to_noise_range(start), nrow))
        prev_image = None
        prev_start = None
        for round_idx in range(1, args.num_rounds + 1):
            with torch.amp.autocast("cuda", enabled=args.enable_amp, dtype=args.amp_dtype):
                image = model.generate(
                    n_samples=n,
                    labels=labels,
                    cfg=args.cfg,
                    args=args,
                    verbose=False,
                    z_t=start,
                )
            rows.append(_row(f"round {round_idx}\nJiT output", _to_image_range(image), nrow))

            round_stats = {
                "round": int(round_idx),
                "start_std": float(start.std().item()),
                "start_mean": float(start.mean().item()),
                "output_std": float(image.std().item()),
                "output_mean": float(image.mean().item()),
            }
            if prev_image is not None:
                round_stats["output_delta_mse"] = _mse(image, prev_image)
                round_stats["output_cosine_to_previous"] = _cosine(image, prev_image)
            if prev_start is not None:
                round_stats["start_delta_mse"] = _mse(start, prev_start)
                round_stats["start_cosine_to_previous"] = _cosine(start, prev_start)
            stats["rounds"].append(round_stats)

            if round_idx < args.num_rounds:
                prev_image = image
                prev_start = start
                start = _sample_reencoded_start(args, encoder, image, args.reencode_mode)
                rows.append(
                    _row(
                        f"round {round_idx}\nDINO({round_idx}) bridge start\nmode={args.reencode_mode}",
                        _to_noise_range(start),
                        nrow,
                    )
                )

    _save_contact(rows, args.out_dir / "dino_recursive_inference_contact.png")
    (args.out_dir / "dino_recursive_inference_stats.json").write_text(
        json.dumps(stats, indent=2),
    )
    print(json.dumps(stats, indent=2))
    print(f"saved {args.out_dir / 'dino_recursive_inference_contact.png'}")


if __name__ == "__main__":
    main()
