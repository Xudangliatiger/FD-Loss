"""Train and visualize a JiT-B VAE-start encoder without JiT post-training.

This diagnostic keeps the JiT generator frozen and trains only the pixel-space
VAE-start encoder used by ``main_jit_vae_start.py``.  The goal is to inspect
whether the learned start distribution is a sampleable Gaussian-like endpoint
or just a lightly noised copy of the image.

Time convention:
    t = 0 is data, t = 1 is the start/noise endpoint.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision.utils import make_grid

from main_fd import average_gradients, get_args_parser
from main_jit_vae_start import (
    VariationalStartEncoder,
    build_paired_loader,
    gaussian_kl_per_dim,
    infinite_loader,
    predict_x0,
)
from utils.builders import create_generation_model
from utils.checkpoint_util import ckpt_resume
from utils.distributed_util import (
    all_reduce_mean,
    broadcast_module_params,
    is_enabled,
    is_main_process,
)
from utils.grad_util import get_grad_norm
from utils.setup_util import setup

logger = logging.getLogger("FD_loss")


def _to_image_range(x: torch.Tensor) -> torch.Tensor:
    return (x.detach().float().clamp(-1, 1) + 1.0) * 0.5


def _to_noise_range(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().float()
    flat = x.flatten(1)
    lo = flat.min(dim=1).values.view(-1, 1, 1, 1)
    hi = flat.max(dim=1).values.view(-1, 1, 1, 1)
    return ((x - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1)


def _row(label: str, images: torch.Tensor, nrow: int, label_width: int = 300) -> Image.Image:
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
    width = max(row.width for row in rows)
    height = sum(row.height for row in rows)
    canvas = Image.new("RGB", (width, height), "white")
    y = 0
    for row in rows:
        canvas.paste(row, (0, y))
        y += row.height
    canvas.save(path)


def cosine_to_x0(z: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
    zf = z.detach().float().flatten(1)
    xf = x0.detach().float().flatten(1)
    return torch.nn.functional.cosine_similarity(zf, xf, dim=1).mean()


def sample_start_from_stats(args, mu: torch.Tensor, logvar: torch.Tensor):
    eps = torch.randn_like(mu)
    if args.vae_start_sample_mode == "posterior":
        sigma = torch.exp(0.5 * logvar)
        return mu + sigma * eps, eps, sigma
    if args.vae_start_sample_mode == "mean_shift":
        sigma = torch.ones_like(mu)
        return eps + args.vae_start_mean_scale * mu, eps, sigma
    raise ValueError(f"unknown VAE-start sample mode: {args.vae_start_sample_mode}")


def reduce_metric(value: float) -> float:
    reduced = all_reduce_mean(torch.tensor(value, device="cuda"))
    return float(reduced.item() if isinstance(reduced, torch.Tensor) else reduced)


@torch.no_grad()
def save_visualization(args, model, encoder, x0, labels, step: int):
    if not is_main_process():
        return

    vis_dir = Path(args.vis_dir) / f"step_{step:07d}"
    vis_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    encoder.eval()

    x0 = x0.cuda(non_blocking=True)
    labels = labels.cuda(non_blocking=True)
    ones = torch.ones(x0.shape[0], device=x0.device)

    torch.manual_seed(args.seed + 100000 + step)
    mu, logvar = encoder.stats(x0)
    z_start, eps_post, sigma = sample_start_from_stats(args, mu, logvar)
    eps_ref = torch.randn_like(mu)
    injected = z_start - eps_post

    vae_recon = predict_x0(model, z_start, ones, labels, drop_labels=False)
    mu_recon = predict_x0(model, mu, ones, labels, drop_labels=False)
    random_recon = predict_x0(model, eps_ref, ones, labels, drop_labels=False)

    stats = {
        "step": int(step),
        "num_images": int(x0.shape[0]),
        "cycle_mse_z_start": float(torch.mean((vae_recon - x0) ** 2).item()),
        "cycle_mse_mu": float(torch.mean((mu_recon - x0) ** 2).item()),
        "random_start_mse": float(torch.mean((random_recon - x0) ** 2).item()),
        "kl_per_dim": float(gaussian_kl_per_dim(mu, logvar).item()),
        "mu_mean": float(mu.mean().item()),
        "mu_std": float(mu.std().item()),
        "mu_x0_cosine": float(cosine_to_x0(mu, x0).item()),
        "logvar_mean": float(logvar.mean().item()),
        "logvar_std": float(logvar.std().item()),
        "sigma_mean": float(sigma.mean().item()),
        "sigma_std": float(sigma.std().item()),
        "z_start_mean": float(z_start.mean().item()),
        "z_start_std": float(z_start.std().item()),
        "z_start_x0_cosine": float(cosine_to_x0(z_start, x0).item()),
        "eps_ref_std": float(eps_ref.std().item()),
        "injected_mean": float(injected.mean().item()),
        "injected_std": float(injected.std().item()),
        "injected_x0_cosine": float(cosine_to_x0(injected, x0).item()),
        "gaussian_stat_gap": float(mu.mean().square().item() + (z_start.std() - 1.0).square().item()),
    }

    nrow = min(8, x0.shape[0])
    rows = [
        _row("real x0", _to_image_range(x0), nrow),
        _row("VAE mean mu(x0)", _to_noise_range(mu), nrow),
        _row("VAE sigma(x0)", _to_noise_range(sigma), nrow),
        _row("posterior eps", _to_noise_range(eps_post), nrow),
        _row("sampled z_start", _to_noise_range(z_start), nrow),
        _row("injected z_start - eps", _to_noise_range(injected), nrow),
        _row("JiT(z_start, t=1)", _to_image_range(vae_recon), nrow),
        _row("JiT(mu, t=1)", _to_image_range(mu_recon), nrow),
        _row("JiT(random eps, t=1)", _to_image_range(random_recon), nrow),
    ]
    _save_contact(rows, vis_dir / "vae_start_contact.png")
    (vis_dir / "vae_start_stats.json").write_text(json.dumps(stats, indent=2))

    latest_contact = Path(args.vis_dir) / "vae_start_contact_latest.png"
    latest_stats = Path(args.vis_dir) / "vae_start_stats_latest.json"
    _save_contact(rows, latest_contact)
    latest_stats.write_text(json.dumps(stats, indent=2))
    logger.info("[VAE-only vis] step=%d stats=%s", step, json.dumps(stats, sort_keys=True))

    model.train(False)
    encoder.train(True)


def save_encoder_checkpoint(args, encoder, optimizer, step: int):
    if not is_main_process():
        return
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    payload = {
        "step": int(step),
        "base_model_ckpt": args.load_from,
        "vae_start_encoder": encoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "vae_start_config": {
            "vae_start_hidden": args.vae_start_hidden,
            "vae_start_kl_weight": args.vae_start_kl_weight,
            "vae_start_cycle_weight": args.vae_start_cycle_weight,
            "vae_start_sample_mode": args.vae_start_sample_mode,
            "vae_start_mean_scale": args.vae_start_mean_scale,
            "vae_start_logvar_min": args.vae_start_logvar_min,
            "vae_start_logvar_max": args.vae_start_logvar_max,
        },
        "model_config": {
            "model": args.model,
            "img_size": args.img_size,
            "num_classes": args.num_classes,
            "rope_2d": args.rope_2d,
            "learned_pe": args.learned_pe,
            "legacy_time_convention": args.legacy_time_convention,
            "ema_type": args.ema_type,
        },
    }
    path = Path(args.ckpt_dir) / f"vae_encoder_step_{step:07d}.pth"
    torch.save(payload, path)
    latest = Path(args.ckpt_dir) / "vae_encoder_latest.pth"
    try:
        latest.unlink()
    except FileNotFoundError:
        pass
    latest.symlink_to(path.name)
    logger.info("[VAE-only] saved encoder checkpoint: %s", path)


def train(args):
    setup(args)
    if args.model != "JiT_B":
        raise ValueError("VAE-only diagnostic currently supports --model JiT_B")

    model, ema_model = create_generation_model(args)
    ckpt_resume(args, model, optimizer=None, model_ema=ema_model)
    del ema_model
    model.eval().requires_grad_(False)

    encoder = VariationalStartEncoder(
        channels=3,
        hidden=args.vae_start_hidden,
        logvar_min=args.vae_start_logvar_min,
        logvar_max=args.vae_start_logvar_max,
    ).cuda()
    if is_enabled():
        broadcast_module_params(encoder, src=0)
    encoder.train().requires_grad_(True)

    loader, sampler = build_paired_loader(args)
    loader_iter = infinite_loader(loader, sampler)
    vis_x0, vis_labels = next(loader_iter)
    vis_x0 = vis_x0[: args.num_vis_images].contiguous()
    vis_labels = vis_labels[: args.num_vis_images].contiguous()

    optimizer = torch.optim.AdamW(
        encoder.parameters(),
        lr=args.vae_start_lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.enable_amp)
    grad_accum_steps = max(1, args.gradient_accumulation_steps)
    total_steps = args.vae_only_steps if args.vae_only_steps > 0 else args.total_steps
    metric_path = Path(args.log_dir) / "vae_only_metrics.json"

    logger.info(
        "[VAE-only] steps=%d global_bsz=%d kl=%.4f cycle=%.4f hidden=%d",
        total_steps,
        args.global_bsz,
        args.vae_start_kl_weight,
        args.vae_start_cycle_weight,
        args.vae_start_hidden,
    )
    save_visualization(args, model, encoder, vis_x0, vis_labels, step=0)

    start_time = time.time()
    for step in range(1, total_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        totals = {
            "loss": 0.0,
            "recon_loss": 0.0,
            "kl_loss": 0.0,
            "start_std": 0.0,
            "mu_std": 0.0,
            "logvar_mean": 0.0,
            "sigma_mean": 0.0,
        }

        for _ in range(grad_accum_steps):
            x0, labels = next(loader_iter)
            x0 = x0.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            ones = torch.ones(x0.shape[0], device=x0.device)

            with torch.amp.autocast("cuda", enabled=args.enable_amp, dtype=args.amp_dtype):
                mu, logvar = encoder.stats(x0)
                z_start, _, sigma = sample_start_from_stats(args, mu, logvar)
                recon = predict_x0(model, z_start, ones, labels, drop_labels=False)
                recon_loss = F.mse_loss(recon, x0)
                kl_loss = gaussian_kl_per_dim(mu, logvar)
                loss = (
                    args.vae_start_cycle_weight * recon_loss
                    + args.vae_start_kl_weight * kl_loss
                )

            scaler.scale(loss / grad_accum_steps).backward()
            totals["loss"] += float(loss.detach())
            totals["recon_loss"] += float(recon_loss.detach())
            totals["kl_loss"] += float(kl_loss.detach())
            totals["start_std"] += float(z_start.std().detach())
            totals["mu_std"] += float(mu.std().detach())
            totals["logvar_mean"] += float(logvar.mean().detach())
            totals["sigma_mean"] += float(sigma.mean().detach())

        scaler.unscale_(optimizer)
        average_gradients(encoder)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            encoder.parameters(),
            args.grad_clip,
        ) if args.grad_clip > 0 else get_grad_norm(encoder.parameters())
        scaler.step(optimizer)
        scaler.update()

        metrics = {k: v / grad_accum_steps for k, v in totals.items()}
        metrics = {k: reduce_metric(v) for k, v in metrics.items()}
        metrics["step"] = step
        metrics["grad_norm"] = float(grad_norm)
        metrics["elapsed_sec"] = time.time() - start_time

        if is_main_process() and (step == 1 or step % args.print_freq == 0):
            with metric_path.open("a") as f:
                f.write(json.dumps(metrics) + "\n")
            logger.info(
                "[VAE-only] step=%d loss=%.6f recon=%.6f kl=%.6f "
                "start_std=%.3f mu_std=%.3f logvar_mean=%.3f sigma_mean=%.3f",
                step,
                metrics["loss"],
                metrics["recon_loss"],
                metrics["kl_loss"],
                metrics["start_std"],
                metrics["mu_std"],
                metrics["logvar_mean"],
                metrics["sigma_mean"],
            )

        if step % args.vae_vis_every == 0 or step == total_steps:
            save_visualization(args, model, encoder, vis_x0, vis_labels, step=step)
        if step % args.vae_save_every == 0 or step == total_steps:
            save_encoder_checkpoint(args, encoder, optimizer, step=step)
        if is_enabled():
            torch.distributed.barrier()

    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        "JiT-B VAE-start encoder-only diagnostic",
        parents=[get_args_parser()],
    )
    parser.add_argument("--image_root", default="", type=str)
    parser.add_argument("--image_list", default="", type=str)
    parser.add_argument("--vae_only_steps", default=5000, type=int)
    parser.add_argument("--vae_vis_every", default=1000, type=int)
    parser.add_argument("--vae_save_every", default=1000, type=int)
    parser.add_argument("--num_vis_images", default=16, type=int)
    parser.add_argument("--vae_start_hidden", default=64, type=int)
    parser.add_argument("--vae_start_lr", default=2e-4, type=float)
    parser.add_argument("--vae_start_kl_weight", default=0.25, type=float)
    parser.add_argument("--vae_start_cycle_weight", default=1.0, type=float)
    parser.add_argument("--vae_start_logvar_min", default=-6.0, type=float)
    parser.add_argument("--vae_start_logvar_max", default=2.0, type=float)
    parser.add_argument("--vae_start_sample_mode", choices=["posterior", "mean_shift"],
                        default="posterior")
    parser.add_argument("--vae_start_mean_scale", default=1.0, type=float)
    return parser


if __name__ == "__main__":
    sys.exit(train(build_parser().parse_args()))
