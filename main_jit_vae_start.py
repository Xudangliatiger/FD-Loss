"""Paired JiT-B VAE-start post-training.

This is the ImageNet-scale version of the Fashion-MNIST VAE-start experiment.
The FD-Loss training entry point is data-free, so this script keeps the paired
recipe separate: read real images, learn an image-conditioned Gaussian start
distribution, and post-train JiT only near the noise/start endpoint.

Time convention in this repository:
    t = 0 is data, t = 1 is the noise/start endpoint.

The user-facing ``--vae_start_tc`` is measured from the start endpoint.  For
example, ``--vae_start_tc 0.30`` uses VAE starts when raw JiT time ``t >= 0.70``.
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from main_fd import average_gradients, get_args_parser
from utils.builders import create_generation_model
from utils.checkpoint_util import AsyncCheckpointSaver, ckpt_resume, save_checkpoint
from utils.data_util import center_crop_arr
from utils.distributed_util import (
    all_reduce_mean,
    broadcast_module_params,
    get_global_rank,
    get_world_size,
    is_enabled,
    is_main_process,
    preempt_requested,
    register_preempt_handler,
)
from utils.ema_util import EMAModel
from utils.grad_util import get_grad_norm
from utils.logging_util import MetricLogger, SmoothedValue
from utils.optimizer_util import create_optimizer
from utils.schedule_util import adjust_learning_rate
from utils.setup_util import setup

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

logger = logging.getLogger("FD_loss")


class ImageListWithLabels(Dataset):
    """ImageNet list file dataset.

    Each line is ``relative/path label``.  ``image_root`` should point to the
    directory containing the class folders, usually ``$IMAGENET_ROOT/train``.
    """

    def __init__(self, list_path: str | os.PathLike, image_root: str | os.PathLike,
                 transform=None):
        self.list_path = Path(list_path)
        self.image_root = Path(image_root)
        self.transform = transform
        self.items: list[tuple[Path, int]] = []
        with self.list_path.open() as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                rel, label = parts[0], int(parts[1])
                path = Path(rel)
                if not path.is_absolute():
                    path = self.image_root / path
                self.items.append((path, label))
        if not self.items:
            raise ValueError(f"no images listed in {self.list_path}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, label = self.items[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class VariationalStartEncoder(nn.Module):
    """Small convolutional encoder producing pixel-space Gaussian starts."""

    def __init__(self, channels: int = 3, hidden: int = 64,
                 logvar_min: float = -6.0, logvar_max: float = 2.0):
        super().__init__()
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max
        groups = min(8, hidden)
        mid = max(hidden // 2, 32)
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1),
            nn.GroupNorm(groups, hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, stride=2, padding=1),
            nn.GroupNorm(groups, hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(groups, hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, stride=2, padding=1),
            nn.GroupNorm(groups, hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(groups, hidden),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(groups, hidden),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden, mid, 3, padding=1),
            nn.GroupNorm(min(8, mid), mid),
            nn.SiLU(),
            nn.Conv2d(mid, channels * 2, 3, padding=1),
        )

    def stats(self, x0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = self.net(x0).chunk(2, dim=1)
        logvar = logvar.clamp(self.logvar_min, self.logvar_max)
        return mu, logvar

    def forward(self, x0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.stats(x0)
        eps = torch.randn_like(mu)
        return mu + torch.exp(0.5 * logvar) * eps, mu, logvar


def build_paired_loader(args):
    transform = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, args.img_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2.0 - 1.0),
    ])

    data_path = Path(args.data_path)
    train_dir = data_path / "train"
    if train_dir.is_dir():
        dataset = datasets.ImageFolder(train_dir, transform=transform)
        logger.info(f"[Data] ImageFolder: {train_dir} ({len(dataset)} images)")
    elif args.image_list and args.image_root:
        dataset = ImageListWithLabels(args.image_list, args.image_root, transform=transform)
        logger.info(f"[Data] Image list: {args.image_list}, root={args.image_root} "
                    f"({len(dataset)} images)")
    else:
        raise FileNotFoundError(
            "paired VAE-start training needs real ImageNet images. Provide either "
            "--data_path with a train/ ImageFolder, or --image_list plus --image_root."
        )

    sampler = DistributedSampler(
        dataset,
        num_replicas=get_world_size(),
        rank=get_global_rank(),
        shuffle=True,
        drop_last=True,
    ) if is_enabled() else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        persistent_workers=args.num_workers > 0,
    )
    return loader, sampler


def infinite_loader(loader, sampler=None):
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


def sample_vae_start(encoder: VariationalStartEncoder, x0: torch.Tensor,
                     mode: str, mean_scale: float):
    mu, logvar = encoder.stats(x0)
    eps = torch.randn_like(mu)
    if mode == "posterior":
        start = mu + torch.exp(0.5 * logvar) * eps
    elif mode == "mean_shift":
        start = eps + mean_scale * mu
    else:
        raise ValueError(f"unknown VAE-start sample mode: {mode}")
    return start, mu, logvar


def gaussian_kl_per_dim(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return 0.5 * (mu.square() + logvar.exp() - logvar - 1.0).mean()


def reduce_float(x) -> float:
    y = all_reduce_mean(x)
    return float(y.item() if isinstance(y, torch.Tensor) else y)


def predict_x0(model, x_t: torch.Tensor, t: torch.Tensor, labels: torch.Tensor,
               *, drop_labels: bool) -> torch.Tensor:
    if t.ndim == 1:
        t = t.view(-1, 1, 1, 1)
    if drop_labels:
        labels = model.drop_labels(labels)
    return model.net(x_t, model._backbone_t(t).flatten(), labels)


def velocity_from_x0(x0: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor,
                     t_eps: float) -> torch.Tensor:
    while t.ndim < x0.ndim:
        t = t[..., None]
    return (x_t - x0) / t.clamp_min(t_eps)


def make_optimizer(args, model, encoder):
    model_params = [p for p in model.parameters() if p.requires_grad]
    enc_params = [p for p in encoder.parameters() if p.requires_grad]
    return torch.optim.AdamW(
        [
            {"params": model_params, "lr": args.lr, "weight_decay": args.weight_decay},
            {"params": enc_params, "lr": args.vae_start_lr, "weight_decay": args.weight_decay},
        ],
        betas=(args.beta1, args.beta2),
    )


def pretrain_encoder(args, model, encoder, loader_iter):
    if args.vae_start_pre_steps <= 0:
        return
    logger.info(f"[VAE-start] pretraining encoder for {args.vae_start_pre_steps} steps")
    model.eval().requires_grad_(False)
    encoder.train().requires_grad_(True)
    opt = torch.optim.AdamW(
        encoder.parameters(),
        lr=args.vae_start_lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.enable_amp)

    for step in range(1, args.vae_start_pre_steps + 1):
        x0, labels = next(loader_iter)
        x0 = x0.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        ones = torch.ones(x0.shape[0], device=x0.device)

        with torch.amp.autocast("cuda", enabled=args.enable_amp, dtype=args.amp_dtype):
            start, mu, logvar = sample_vae_start(
                encoder, x0, args.vae_start_sample_mode, args.vae_start_mean_scale,
            )
            recon = predict_x0(model, start, ones, labels, drop_labels=False)
            recon_loss = F.mse_loss(recon, x0)
            kl_loss = gaussian_kl_per_dim(mu, logvar)
            loss = args.vae_start_cycle_weight * recon_loss + args.vae_start_kl_weight * kl_loss

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        average_gradients(encoder)
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.grad_clip)
        scaler.step(opt)
        scaler.update()

        if step == 1 or step % args.print_freq == 0:
            logger.info(
                "[VAE-start pre] step=%d loss=%.6f recon=%.6f kl=%.6f "
                "start_std=%.3f mu_std=%.3f logvar_mean=%.3f",
                step,
                reduce_float(loss.detach()),
                reduce_float(recon_loss.detach()),
                reduce_float(kl_loss.detach()),
                reduce_float(start.std().detach()),
                reduce_float(mu.std().detach()),
                reduce_float(logvar.mean().detach()),
            )


def train_post(args):
    setup(args)
    register_preempt_handler()

    model, ema_model = create_generation_model(args)
    ckpt_resume(args, model, optimizer=None, model_ema=ema_model)

    if args.model != "JiT_B":
        raise ValueError("main_jit_vae_start.py currently supports --model JiT_B only")

    loader, sampler = build_paired_loader(args)
    loader_iter = infinite_loader(loader, sampler)

    encoder = VariationalStartEncoder(
        channels=3,
        hidden=args.vae_start_hidden,
        logvar_min=args.vae_start_logvar_min,
        logvar_max=args.vae_start_logvar_max,
    ).cuda()
    if is_enabled():
        broadcast_module_params(encoder, src=0)

    pretrain_encoder(args, model, encoder, loader_iter)

    model.train().requires_grad_(True)
    encoder.train().requires_grad_(True)
    optimizer = make_optimizer(args, model, encoder)
    saver = AsyncCheckpointSaver()
    scaler = torch.amp.GradScaler("cuda", enabled=args.enable_amp)

    metric_file = os.path.join(args.log_dir, "training_metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metric_file, prefetch=True)
    for name, window, fmt in [
        ("lr", 1, "{value:.6f}"),
        ("samples/s/device", args.print_freq, "{avg:.2f}"),
        ("samples/s", args.print_freq, "{avg:.2f}"),
        ("samples_seen(M)", args.print_freq, "{value:.2f}"),
        ("device_mem(GB)", args.print_freq, "{value:.2f}"),
    ]:
        metric_logger.add_meter(name, SmoothedValue(window, fmt))

    logger.info(
        "[VAE-start] post-training raw JiT t in [%.3f, %.3f], "
        "VAE-start where distance-from-start < %.3f (raw t >= %.3f)",
        args.train_t_min,
        args.train_t_max,
        args.vae_start_tc,
        1.0 - args.vae_start_tc,
    )
    logger.info("training from step %d -> %d", args.current_step, args.total_steps)

    session_start = time.time()
    step_start = time.perf_counter()
    last_ckpt_step = args.current_step

    def _save(step):
        elapsed = time.time() - session_start + args.last_elapsed_time
        extra = {
            "vae_start_encoder": encoder.state_dict(),
            "vae_start_config": {
                "vae_start_tc": args.vae_start_tc,
                "vae_start_kl_weight": args.vae_start_kl_weight,
                "vae_start_cycle_weight": args.vae_start_cycle_weight,
                "vae_start_sample_mode": args.vae_start_sample_mode,
                "vae_start_mean_scale": args.vae_start_mean_scale,
                "train_t_min": args.train_t_min,
                "train_t_max": args.train_t_max,
            },
        }
        save_checkpoint(args, step, model, optimizer, ema_model, elapsed,
                        saver=saver, extra=extra)
        if is_enabled():
            torch.distributed.barrier()

    for step, _ in metric_logger.log_every(
        iter(int, 1), args.print_freq, header="VAE-start:",
        start_iteration=args.current_step, n_iterations=args.total_steps,
    ):
        adjust_learning_rate(optimizer, step, args)
        x0, labels = next(loader_iter)
        x0 = x0.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        t = args.train_t_min + (args.train_t_max - args.train_t_min) * torch.rand(
            x0.shape[0], device=x0.device,
        )
        t_view = t.view(-1, 1, 1, 1)
        eps = torch.randn_like(x0) * args.noise_scale

        with torch.amp.autocast("cuda", enabled=args.enable_amp, dtype=args.amp_dtype):
            vae_start, mu, logvar = sample_vae_start(
                encoder, x0, args.vae_start_sample_mode, args.vae_start_mean_scale,
            )
            # Raw JiT t=1 is the start endpoint.  Use VAE starts only near it.
            vae_mask = (t >= (1.0 - args.vae_start_tc)).float().view(-1, 1, 1, 1)
            start = vae_mask * vae_start + (1.0 - vae_mask) * eps
            x_t = (1.0 - t_view) * x0 + t_view * start

            x0_pred = predict_x0(model, x_t, t, labels, drop_labels=True)
            v_pred = velocity_from_x0(x0_pred, x_t, t, args.t_eps)
            v_true = velocity_from_x0(x0, x_t, t, args.t_eps)
            jit_loss = F.mse_loss(v_pred, v_true)

            ones = torch.ones_like(t)
            cycle_pred = predict_x0(model, vae_start, ones, labels, drop_labels=False)
            cycle_loss = F.mse_loss(cycle_pred, x0)
            kl_loss = gaussian_kl_per_dim(mu, logvar)
            loss = (
                jit_loss
                + args.vae_start_cycle_weight * cycle_loss
                + args.vae_start_kl_weight * kl_loss
            )

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        average_gradients(model)
        average_gradients(encoder)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(encoder.parameters()),
            args.grad_clip,
        ) if args.grad_clip > 0 else get_grad_norm(
            list(model.parameters()) + list(encoder.parameters())
        )
        if torch.isfinite(grad_norm):
            scaler.step(optimizer)
            ema_model.step(model)
        else:
            logger.warning("[step %d] NaN/Inf grad_norm; skipping update", step)
        scaler.update()
        torch.cuda.synchronize()

        args.current_step = step + 1
        args.samples_seen += args.global_bsz

        step_time = time.perf_counter() - step_start
        step_start = time.perf_counter()
        sps = args.batch_size / step_time if step_time > 0 else 0.0
        mem_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)
        metric_logger.update(
            loss=reduce_float(loss.detach()),
            jit_loss=reduce_float(jit_loss.detach()),
            cycle_loss=reduce_float(cycle_loss.detach()),
            kl_loss=reduce_float(kl_loss.detach()),
            vae_frac=reduce_float(vae_mask.mean().detach()),
            start_std=reduce_float(vae_start.std().detach()),
            mu_std=reduce_float(mu.std().detach()),
            logvar_mean=reduce_float(logvar.mean().detach()),
            grad_norm=grad_norm,
            lr=optimizer.param_groups[0]["lr"],
            **{
                "samples/s/device": sps,
                "samples/s": sps * args.world_size,
                "samples_seen(M)": args.samples_seen / 1e6,
                "device_mem(GB)": mem_gb,
            },
        )

        if (args.current_step - last_ckpt_step >= args.save_every
                or args.current_step == args.total_steps):
            _save(step)
            last_ckpt_step = args.current_step

        if args.milestone_every > 0 and step > 0 and step % args.milestone_every == 0:
            _save(step)

        if preempt_requested():
            logger.info("preemption requested at step %d; saving checkpoint", args.current_step)
            saver.wait()
            _save(step)
            return 0

    saver.wait()
    metric_logger.synchronize_between_processes()
    total = time.time() - session_start + args.last_elapsed_time
    logger.info("averaged stats: %s", metric_logger)
    logger.info("Training complete. Total time: %s", datetime.timedelta(seconds=int(total)))
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        "JiT VAE-start paired post-training", parents=[get_args_parser()],
    )
    parser.add_argument("--image_root", default="", type=str,
                        help="root for paths in --image_list, usually ImageNet/train")
    parser.add_argument("--image_list", default="", type=str,
                        help="optional list file with 'relative_path label' rows")
    parser.add_argument("--train_t_min", default=0.02, type=float)
    parser.add_argument("--train_t_max", default=1.0, type=float)
    parser.add_argument("--vae_start_tc", default=0.30, type=float,
                        help="distance from the start endpoint where VAE starts are used")
    parser.add_argument("--vae_start_pre_steps", default=2000, type=int)
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
    args = build_parser().parse_args()
    sys.exit(train_post(args))
