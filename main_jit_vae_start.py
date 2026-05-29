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
import glob
import io
import json
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
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset, DistributedSampler, IterableDataset, get_worker_info
from torchvision.utils import make_grid

from main_fd import average_gradients, get_args_parser
from frechet_distance.judges import (
    extract_judge_features,
    fill_all_queues,
    resolve_per_model_args,
    run_sanity_check,
    save_fd_queue_states,
)
from frechet_distance.losses import (
    compute_frechet_distance_loss,
    diff_all_gather,
    load_mu_and_sigma_reference,
    precompute_sigma_ref_sqrt,
)
from frechet_distance.queue import FeatureQueue
from frechet_distance.repr_models import load_repr_model, model_short_name
from models.dinov2_start import DINOv2LatentStartEncoder, DINOv2SphereStartEncoder
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
from utils.start_util import apply_start_support, sample_start

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


class HfParquetImageNetDataset(IterableDataset):
    """Stream HuggingFace ImageNet parquet shards as ``(image_tensor, label)``."""

    def __init__(self, files, img_size: int):
        super().__init__()
        self.files = list(files)
        self.img_size = img_size
        self.to_tensor = transforms.ToTensor()

    def _iter_files_for_worker(self):
        worker = get_worker_info()
        if worker is None:
            return self.files
        return self.files[worker.id::worker.num_workers]

    def __iter__(self):
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError(
                "Reading HuggingFace parquet ImageNet requires pyarrow. "
                "Install pyarrow in the active environment."
            ) from exc

        for path in self._iter_files_for_worker():
            table = pq.read_table(path)
            names = set(table.column_names)
            image_col = "image" if "image" in names else None
            label_col = "label" if "label" in names else ("labels" if "labels" in names else None)
            if image_col is None or label_col is None:
                raise ValueError(f"Could not find image/label columns in {path}: {table.column_names}")

            images = table[image_col].to_pylist()
            labels = table[label_col].to_pylist()
            for image_obj, label in zip(images, labels):
                if isinstance(image_obj, dict):
                    if image_obj.get("bytes") is not None:
                        image = Image.open(io.BytesIO(image_obj["bytes"]))
                    elif image_obj.get("path") is not None:
                        image = Image.open(image_obj["path"])
                    else:
                        raise ValueError(f"Unsupported HF image object keys: {list(image_obj.keys())}")
                elif isinstance(image_obj, (bytes, bytearray)):
                    image = Image.open(io.BytesIO(image_obj))
                else:
                    raise TypeError(f"Unsupported image object type: {type(image_obj)}")

                image = image.convert("RGB")
                image = center_crop_arr(image, self.img_size)
                yield self.to_tensor(image) * 2.0 - 1.0, int(label)


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


def build_vae_start_encoder(args) -> nn.Module:
    if args.vae_start_encoder_type == "conv":
        return VariationalStartEncoder(
            channels=3,
            hidden=args.vae_start_hidden,
            logvar_min=args.vae_start_logvar_min,
            logvar_max=args.vae_start_logvar_max,
        )
    if args.vae_start_encoder_type == "dinov2_latent":
        return DINOv2LatentStartEncoder(
            channels=3,
            img_size=args.img_size,
            patch_size=args.dinov2_start_patch_size,
            model_name=args.dinov2_start_model,
            num_latent_tokens=args.dinov2_start_latent_tokens,
            token_dim=args.dinov2_start_token_dim,
            pretrained=not args.dinov2_start_no_pretrained,
            pretrained_path=args.dinov2_start_pretrained_path,
            freeze_backbone=not args.dinov2_start_train_backbone,
            logvar_min=args.vae_start_logvar_min,
            logvar_max=args.vae_start_logvar_max,
        )
    if args.vae_start_encoder_type == "dinov2_sphere":
        return DINOv2SphereStartEncoder(
            channels=3,
            img_size=args.img_size,
            patch_size=args.dinov2_start_patch_size,
            model_name=args.dinov2_start_model,
            num_latent_tokens=args.dinov2_start_latent_tokens,
            pretrained=not args.dinov2_start_no_pretrained,
            pretrained_path=args.dinov2_start_pretrained_path,
            freeze_encoder_backbone=not args.dinov2_start_train_backbone,
            freeze_decoder_backbone=args.dinov2_start_freeze_decoder_backbone,
            noise_sigma_max_angle=args.dinov2_start_noise_angle,
        )
    raise ValueError(f"unknown VAE-start encoder type: {args.vae_start_encoder_type}")


def vae_start_config_dict(args) -> dict:
    return {
        "vae_start_encoder_type": args.vae_start_encoder_type,
        "vae_start_hidden": args.vae_start_hidden,
        "vae_start_kl_weight": args.vae_start_kl_weight,
        "vae_start_cycle_weight": args.vae_start_cycle_weight,
        "vae_start_sample_mode": args.vae_start_sample_mode,
        "vae_start_mean_scale": args.vae_start_mean_scale,
        "vae_start_logvar_min": args.vae_start_logvar_min,
        "vae_start_logvar_max": args.vae_start_logvar_max,
        "dinov2_start_model": args.dinov2_start_model,
        "dinov2_start_patch_size": args.dinov2_start_patch_size,
        "dinov2_start_latent_tokens": args.dinov2_start_latent_tokens,
        "dinov2_start_token_dim": args.dinov2_start_token_dim,
        "dinov2_start_train_backbone": args.dinov2_start_train_backbone,
        "dinov2_start_freeze_decoder_backbone": args.dinov2_start_freeze_decoder_backbone,
        "dinov2_start_no_pretrained": args.dinov2_start_no_pretrained,
        "dinov2_start_pretrained_path": args.dinov2_start_pretrained_path,
        "dinov2_start_noise_angle": args.dinov2_start_noise_angle,
        "start_support_mode": args.start_support_mode,
    }


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
    elif (parquet_files := sorted(glob.glob(str(data_path / "data" / "train-*.parquet")))):
        rank_files = parquet_files[get_global_rank()::get_world_size()]
        dataset = HfParquetImageNetDataset(rank_files, args.img_size)
        logger.info(f"[Data] HF parquet ImageNet: {data_path} "
                    f"({len(parquet_files)} shards, {len(rank_files)} on this rank)")
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            persistent_workers=args.num_workers > 0,
        )
        return loader, None
    else:
        raise FileNotFoundError(
            "paired VAE-start training needs real ImageNet images. Provide either "
            "--data_path with a train/ ImageFolder or HF parquet data/train-*.parquet, "
            "or --image_list plus --image_root."
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


def sample_vae_start(encoder: nn.Module, x0: torch.Tensor,
                     mode: str, mean_scale: float):
    if getattr(encoder, "start_kind", None) == "sphere_latent":
        sample = encoder.sample_start(x0)
        encoder._last_start_sample = sample
        return sample["start"], sample["clean_start"], torch.zeros_like(sample["clean_start"])
    mu, logvar = encoder.stats(x0)
    eps = torch.randn_like(mu)
    if mode == "posterior":
        start = mu + torch.exp(0.5 * logvar) * eps
    elif mode == "mean_shift":
        start = eps + mean_scale * mu
    else:
        raise ValueError(f"unknown VAE-start sample mode: {mode}")
    return start, mu, logvar


def start_regularization_loss(encoder: nn.Module, mu: torch.Tensor,
                              logvar: torch.Tensor) -> torch.Tensor:
    if getattr(encoder, "start_kind", None) == "sphere_latent":
        return torch.zeros((), device=mu.device, dtype=mu.dtype)
    return gaussian_kl_per_dim(mu, logvar)


def _to_image_range(x: torch.Tensor) -> torch.Tensor:
    return (x.detach().float().clamp(-1, 1) + 1.0) * 0.5


def _to_noise_range(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().float()
    flat = x.flatten(1)
    lo = flat.min(dim=1).values.view(-1, 1, 1, 1)
    hi = flat.max(dim=1).values.view(-1, 1, 1, 1)
    return ((x - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1)


def _row(label: str, images: torch.Tensor, nrow: int,
         label_width: int = 280) -> Image.Image:
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
    path.parent.mkdir(parents=True, exist_ok=True)
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
    return F.cosine_similarity(zf, xf, dim=1).mean()


def load_vae_start_encoder(args, encoder: nn.Module):
    if not args.vae_start_encoder_ckpt:
        return
    ckpt_path = Path(args.vae_start_encoder_ckpt)
    payload = torch.load(ckpt_path, map_location="cpu")
    if isinstance(payload, dict) and "vae_start_encoder" in payload:
        state = payload["vae_start_encoder"]
    elif isinstance(payload, dict) and "state_dict" in payload:
        state = payload["state_dict"]
    else:
        state = payload
    encoder.load_state_dict(state, strict=True)
    logger.info("[VAE-start] loaded encoder checkpoint: %s", ckpt_path)


@torch.no_grad()
def save_post_visualization(args, model, encoder, x0, labels, step: int):
    if not is_main_process() or encoder is None:
        return

    was_training = model.training
    encoder_was_training = encoder.training
    model.eval()
    encoder.eval()

    vis_dir = Path(args.vis_dir) / "vae_start_post_vis" / f"step_{step:07d}"
    x0 = x0.cuda(non_blocking=True)
    labels = labels.cuda(non_blocking=True)
    ones = torch.ones(x0.shape[0], device=x0.device)

    torch.manual_seed(args.seed + 300000 + step)
    vae_start, mu, logvar = sample_vae_start(
        encoder, x0, args.vae_start_sample_mode, args.vae_start_mean_scale,
    )
    vae_start = apply_start_support(
        vae_start, mode=args.start_support_mode, noise_scale=args.noise_scale,
    )
    mu_start = apply_start_support(
        mu, mode=args.start_support_mode, noise_scale=args.noise_scale,
    )
    random_start = sample_start(
        tuple(x0.shape), device=x0.device, dtype=x0.dtype,
        noise_scale=args.noise_scale, mode=args.start_support_mode,
    )
    random_labels = torch.randint(0, args.num_classes, labels.shape, device=labels.device)

    vae_recon = predict_x0(model, vae_start, ones, labels, drop_labels=False)
    mu_recon = predict_x0(model, mu_start, ones, labels, drop_labels=False)
    random_one_step = predict_x0(model, random_start, ones, random_labels, drop_labels=False)
    sigma = torch.exp(0.5 * logvar)
    reg_loss = start_regularization_loss(encoder, mu, logvar)

    stats = {
        "step": int(step),
        "num_images": int(x0.shape[0]),
        "cycle_mse_z_start": float(torch.mean((vae_recon - x0) ** 2).item()),
        "cycle_mse_mu": float(torch.mean((mu_recon - x0) ** 2).item()),
        "random_one_step_std": float(random_one_step.std().item()),
        "kl_per_dim": float(reg_loss.item()),
        "mu_mean": float(mu.mean().item()),
        "mu_std": float(mu.std().item()),
        "mu_x0_cosine": float(cosine_to_x0(mu, x0).item()),
        "sigma_mean": float(sigma.mean().item()),
        "sigma_std": float(sigma.std().item()),
        "z_start_mean": float(vae_start.mean().item()),
        "z_start_std": float(vae_start.std().item()),
        "z_start_x0_cosine": float(cosine_to_x0(vae_start, x0).item()),
        "random_start_std": float(random_start.std().item()),
        "start_support_mode": args.start_support_mode,
        "vae_start_encoder_ckpt": args.vae_start_encoder_ckpt,
        "freeze_vae_start_encoder": bool(args.freeze_vae_start_encoder),
    }
    aux = getattr(encoder, "_last_start_sample", None)
    if isinstance(aux, dict):
        stats.update({
            "latent_clean_std": float(aux["latent_clean"].std().item()),
            "latent_noisy_std": float(aux["latent_noisy"].std().item()),
            "latent_cosine": float(aux["latent_cosine"].item()),
            "latent_radius_mean": float(aux["latent_radius"].mean().item()),
            "random_bridge_std": float(aux["random_start"].std().item()),
        })

    nrow = min(8, x0.shape[0])
    rows = [
        _row("real x0", _to_image_range(x0), nrow),
        _row("VAE mean mu(x0)", _to_noise_range(mu), nrow),
        _row("VAE sigma(x0)", _to_noise_range(sigma), nrow),
        _row(f"{args.start_support_mode} z_start", _to_noise_range(vae_start), nrow),
        _row("JiT(VAE z_start, t=1)", _to_image_range(vae_recon), nrow),
        _row("JiT(supported mu, t=1)", _to_image_range(mu_recon), nrow),
        _row(f"random {args.start_support_mode} start", _to_noise_range(random_start), nrow),
        _row("JiT(random start, t=1)", _to_image_range(random_one_step), nrow),
    ]
    _save_contact(rows, vis_dir / "vae_start_post_contact.png")
    (vis_dir / "vae_start_post_stats.json").write_text(json.dumps(stats, indent=2))

    latest_dir = Path(args.vis_dir) / "vae_start_post_vis"
    _save_contact(rows, latest_dir / "vae_start_post_contact_latest.png")
    (latest_dir / "vae_start_post_stats_latest.json").write_text(json.dumps(stats, indent=2))
    logger.info("[VAE-start post vis] step=%d stats=%s", step, json.dumps(stats, sort_keys=True))

    model.train(was_training)
    encoder.train(encoder_was_training)


def shuffle_vae_stats(mu: torch.Tensor, logvar: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Break image-start pairing while preserving the batch marginal start prior."""
    if mu.shape[0] <= 1:
        return mu, logvar
    perm = torch.randperm(mu.shape[0], device=mu.device)
    return mu[perm], logvar[perm]


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


def make_optimizer(args, model, encoder=None):
    model_params = [p for p in model.parameters() if p.requires_grad]
    param_groups = [
        {"params": model_params, "lr": args.lr, "weight_decay": args.weight_decay},
    ]
    if encoder is not None:
        enc_params = [p for p in encoder.parameters() if p.requires_grad]
        if enc_params:
            param_groups.append(
                {"params": enc_params, "lr": args.vae_start_lr, "weight_decay": args.weight_decay},
            )
    return torch.optim.AdamW(param_groups, betas=(args.beta1, args.beta2))


def build_fd_judges(args, model):
    if args.vae_start_fd_weight <= 0.0:
        return []

    args.input_channels = model.in_channels
    args.input_size = model.input_size
    resolve_per_model_args(args)

    judges = []
    for name, stats_path, weight, pool_type, target_size in zip(
        args.fd_repr_models, args.fd_repr_stats_paths,
        args.fd_repr_weights, args.fd_repr_pool_types,
        args.fd_target_sizes,
    ):
        if name.startswith("self_"):
            raise ValueError("VAE-start + FD currently supports external FD judges only")
        repr_model, feat_dim, _, _ = load_repr_model(name, target_size=target_size)
        repr_model.eval().requires_grad_(False)
        mu_ref, sigma_ref = load_mu_and_sigma_reference(stats_path, pool_type=pool_type)
        queue = FeatureQueue(
            size=args.queue_size,
            feat_dim=feat_dim,
            online_accum=args.fd_online_accum,
            ema_beta=args.fd_ema_beta,
        ).cuda()
        sigma_ref_sqrt = precompute_sigma_ref_sqrt(sigma_ref) if args.fd_eigvalsh else None
        short = model_short_name(name)
        judges.append({
            "name": short,
            "model": repr_model,
            "feat_dim": feat_dim,
            "pool_type": pool_type,
            "mu_ref": mu_ref,
            "sigma_ref": sigma_ref,
            "sigma_ref_sqrt": sigma_ref_sqrt,
            "queue": queue,
            "weight": weight,
            "input_range": "zero_one",
            "requires_labels": False,
        })
        stats_mode = f"ema(beta={args.fd_ema_beta})" if args.fd_ema_beta > 0 else (
            "online_accum" if args.fd_online_accum else "snapshot"
        )
        logger.info(
            "[VAE-start FD] Repr '%s' (%s): feat_dim=%d weight=%.4f pool=%s "
            "stats=%s stats_mode=%s eigvalsh=%s",
            short, name, feat_dim, weight, pool_type, stats_path, stats_mode, args.fd_eigvalsh,
        )

    logger.info("[VAE-start FD] Filling %d feature queue(s)", len(judges))
    fill_all_queues(judges, model, args)
    run_sanity_check(judges, args.queue_size, args=args)
    return judges


def compute_fd_branch_loss(args, model, judges):
    if not judges:
        return torch.zeros((), device="cuda"), {}, []

    input_shape = (model.in_channels, model.input_size, model.input_size)
    z = sample_start(
        (args.batch_size, *input_shape),
        device="cuda",
        noise_scale=args.noise_scale,
        mode=args.start_support_mode,
    )
    labels = torch.randint(0, args.num_classes, (args.batch_size,), device="cuda")
    sampling_args = {
        "t_min": args.interval_min,
        "t_max": args.interval_max,
        "cfg": args.cfg,
        "num_steps": args.num_sampling_steps,
    }
    sampled_model_range = model.sample_images_with_grad(z, labels, sampling_args=sampling_args)
    sampled_01 = sampled_model_range * 0.5 + 0.5

    all_new_feats = []
    fd_loss = torch.zeros((), device="cuda")
    fd_metrics = {}
    for judge in judges:
        feats = extract_judge_features(judge, sampled_01, labels=labels)
        new_feats = diff_all_gather(feats)
        all_new_feats.append((judge, new_feats))

        sigma_ref_sqrt = judge.get("sigma_ref_sqrt")
        if judge["queue"].online_accum or judge["queue"].ema_stats:
            mu, sigma = judge["queue"].build_feats_stats(new_feats)
            fid = compute_frechet_distance_loss(
                judge["mu_ref"], judge["sigma_ref"],
                mu=mu, sigma=sigma, sigma_ref_sqrt=sigma_ref_sqrt,
            )
        else:
            all_feats = judge["queue"].build_feats_snapshot(new_feats)
            fid = compute_frechet_distance_loss(
                judge["mu_ref"], judge["sigma_ref"],
                all_feats=all_feats, sigma_ref_sqrt=sigma_ref_sqrt,
            )
        normalized = fid / (fid.detach() + args.fd_fid_norm_eps)
        fd_loss = fd_loss + judge["weight"] * normalized
        fd_metrics[f"fid_{judge['name']}"] = float(fid.detach())

    return fd_loss, fd_metrics, all_new_feats


def pretrain_encoder(args, model, encoder, loader_iter):
    if args.vae_start_pre_steps <= 0:
        return
    grad_accum_steps = max(1, args.gradient_accumulation_steps)
    logger.info(
        "[VAE-start] pretraining encoder for %d steps "
        "(micro_batch_per_gpu=%d, world_size=%d, gradient_accumulation_steps=%d, "
        "effective_global_batch=%d)",
        args.vae_start_pre_steps,
        args.batch_size,
        args.world_size,
        grad_accum_steps,
        args.global_bsz,
    )
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
        opt.zero_grad(set_to_none=True)
        metric_totals = {
            "loss": 0.0,
            "recon_loss": 0.0,
            "kl_loss": 0.0,
            "start_std": 0.0,
            "mu_std": 0.0,
            "logvar_mean": 0.0,
        }
        for _ in range(grad_accum_steps):
            x0, labels = next(loader_iter)
            x0 = x0.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            ones = torch.ones(x0.shape[0], device=x0.device)

            with torch.amp.autocast("cuda", enabled=args.enable_amp, dtype=args.amp_dtype):
                start, mu, logvar = sample_vae_start(
                    encoder, x0, args.vae_start_sample_mode, args.vae_start_mean_scale,
                )
                start = apply_start_support(
                    start, mode=args.start_support_mode, noise_scale=args.noise_scale,
                )
                recon = predict_x0(model, start, ones, labels, drop_labels=False)
                recon_loss = F.mse_loss(recon, x0)
                kl_loss = start_regularization_loss(encoder, mu, logvar)
                loss = (
                    args.vae_start_cycle_weight * recon_loss
                    + args.vae_start_kl_weight * kl_loss
                )

            scaler.scale(loss / grad_accum_steps).backward()
            metric_totals["loss"] += float(loss.detach())
            metric_totals["recon_loss"] += float(recon_loss.detach())
            metric_totals["kl_loss"] += float(kl_loss.detach())
            metric_totals["start_std"] += float(start.std().detach())
            metric_totals["mu_std"] += float(mu.std().detach())
            metric_totals["logvar_mean"] += float(logvar.mean().detach())

        scaler.unscale_(opt)
        average_gradients(encoder)
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.grad_clip)
        scaler.step(opt)
        scaler.update()

        if step == 1 or step % args.print_freq == 0:
            metrics = {k: v / grad_accum_steps for k, v in metric_totals.items()}
            logger.info(
                "[VAE-start pre] step=%d loss=%.6f recon=%.6f kl=%.6f "
                "start_std=%.3f mu_std=%.3f logvar_mean=%.3f",
                step,
                reduce_float(torch.tensor(metrics["loss"], device="cuda")),
                reduce_float(torch.tensor(metrics["recon_loss"], device="cuda")),
                reduce_float(torch.tensor(metrics["kl_loss"], device="cuda")),
                reduce_float(torch.tensor(metrics["start_std"], device="cuda")),
                reduce_float(torch.tensor(metrics["mu_std"], device="cuda")),
                reduce_float(torch.tensor(metrics["logvar_mean"], device="cuda")),
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
    vis_x0, vis_labels = next(loader_iter)
    vis_x0 = vis_x0[: args.vae_start_vis_images].contiguous()
    vis_labels = vis_labels[: args.vae_start_vis_images].contiguous()

    use_vae_start = args.vae_start_ablation_mode == "vae_start"
    encoder = None
    if use_vae_start:
        encoder = build_vae_start_encoder(args).cuda()
        if is_enabled():
            broadcast_module_params(encoder, src=0)
        load_vae_start_encoder(args, encoder)
        if args.freeze_vae_start_encoder:
            encoder.eval().requires_grad_(False)
            logger.info("[VAE-start] encoder is frozen for post-training")
        else:
            pretrain_encoder(args, model, encoder, loader_iter)

    model.train().requires_grad_(True)
    if encoder is not None:
        if args.freeze_vae_start_encoder:
            encoder.eval().requires_grad_(False)
        else:
            encoder.train().requires_grad_(True)
    fd_judges = build_fd_judges(args, model)
    model.train().requires_grad_(True)
    if encoder is not None:
        if args.freeze_vae_start_encoder:
            encoder.eval().requires_grad_(False)
        else:
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
        "[VAE-start] ablation_mode=%s raw JiT t in [%.3f, %.3f], "
        "VAE-start where distance-from-start < %.3f (raw t >= %.3f)",
        args.vae_start_ablation_mode,
        args.train_t_min,
        args.train_t_max,
        args.vae_start_tc,
        1.0 - args.vae_start_tc,
    )
    logger.info("training from step %d -> %d", args.current_step, args.total_steps)
    grad_accum_steps = max(1, args.gradient_accumulation_steps)
    logger.info(
        "effective global batch size: %d "
        "(micro_batch_per_gpu=%d, world_size=%d, gradient_accumulation_steps=%d)",
        args.global_bsz,
        args.batch_size,
        args.world_size,
        grad_accum_steps,
    )

    session_start = time.time()
    step_start = time.perf_counter()
    last_ckpt_step = args.current_step

    def _save(step):
        elapsed = time.time() - session_start + args.last_elapsed_time
        extra = {
            "vae_start_config": {
                "vae_start_ablation_mode": args.vae_start_ablation_mode,
                "vae_start_tc": args.vae_start_tc,
                "train_t_min": args.train_t_min,
                "train_t_max": args.train_t_max,
                "vae_start_fd_weight": args.vae_start_fd_weight,
                "vae_start_encoder_ckpt": args.vae_start_encoder_ckpt,
                "freeze_vae_start_encoder": args.freeze_vae_start_encoder,
                **vae_start_config_dict(args),
            },
        }
        if encoder is not None:
            extra["vae_start_encoder"] = encoder.state_dict()
        if fd_judges:
            extra["fd_queue_states"] = save_fd_queue_states(fd_judges)
        save_checkpoint(args, step, model, optimizer, ema_model, elapsed,
                        saver=saver, extra=extra)
        if is_enabled():
            torch.distributed.barrier()

    if args.vae_start_vis_every > 0 and encoder is not None:
        save_post_visualization(args, model, encoder, vis_x0, vis_labels,
                                step=args.current_step)

    for step, _ in metric_logger.log_every(
        iter(int, 1), args.print_freq, header="VAE-start:",
        start_iteration=args.current_step, n_iterations=args.total_steps,
    ):
        adjust_learning_rate(optimizer, step, args)
        optimizer.zero_grad(set_to_none=True)
        metric_totals = {
            "loss": 0.0,
            "jit_loss": 0.0,
            "cycle_loss": 0.0,
            "kl_loss": 0.0,
            "fd_loss": 0.0,
            "vae_frac": 0.0,
            "start_std": 0.0,
            "mu_std": 0.0,
            "logvar_mean": 0.0,
        }
        fd_metric_totals = {}
        for _ in range(grad_accum_steps):
            x0, labels = next(loader_iter)
            x0 = x0.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            t = args.train_t_min + (args.train_t_max - args.train_t_min) * torch.rand(
                x0.shape[0], device=x0.device,
            )
            t_view = t.view(-1, 1, 1, 1)
            eps = sample_start(
                tuple(x0.shape),
                device=x0.device,
                dtype=x0.dtype,
                noise_scale=args.noise_scale,
                mode=args.start_support_mode,
            )

            with torch.amp.autocast("cuda", enabled=args.enable_amp, dtype=args.amp_dtype):
                if use_vae_start:
                    vae_start, mu, logvar = sample_vae_start(
                        encoder, x0, args.vae_start_sample_mode, args.vae_start_mean_scale,
                    )
                    vae_start = apply_start_support(
                        vae_start, mode=args.start_support_mode, noise_scale=args.noise_scale,
                    )
                    # Raw JiT t=1 is the start endpoint.  Use VAE starts only near it.
                    vae_mask = (t >= (1.0 - args.vae_start_tc)).float().view(-1, 1, 1, 1)
                    start = vae_mask * vae_start + (1.0 - vae_mask) * eps
                else:
                    start = eps
                    vae_start = eps
                    mu = torch.zeros_like(eps)
                    logvar = torch.zeros_like(eps)
                    vae_mask = torch.zeros_like(t_view)
                x_t = (1.0 - t_view) * x0 + t_view * start

                x0_pred = predict_x0(model, x_t, t, labels, drop_labels=True)
                v_pred = velocity_from_x0(x0_pred, x_t, t, args.t_eps)
                v_true = velocity_from_x0(x0, x_t, t, args.t_eps)
                jit_loss = F.mse_loss(v_pred, v_true)

                if use_vae_start:
                    ones = torch.ones_like(t)
                    cycle_pred = predict_x0(model, vae_start, ones, labels, drop_labels=False)
                    cycle_loss = F.mse_loss(cycle_pred, x0)
                    kl_loss = start_regularization_loss(encoder, mu, logvar)
                    paired_loss = (
                        jit_loss
                        + args.vae_start_cycle_weight * cycle_loss
                        + args.vae_start_kl_weight * kl_loss
                    )
                else:
                    cycle_loss = torch.zeros((), device=x0.device)
                    kl_loss = torch.zeros((), device=x0.device)
                    paired_loss = jit_loss

            fd_loss = torch.zeros((), device=x0.device)
            fd_metrics = {}
            fd_new_feats = []
            if fd_judges:
                with torch.amp.autocast("cuda", enabled=False):
                    fd_loss, fd_metrics, fd_new_feats = compute_fd_branch_loss(
                        args, model, fd_judges,
                    )

            loss = paired_loss + args.vae_start_fd_weight * fd_loss

            scaler.scale(loss / grad_accum_steps).backward()
            for judge, new_feats in fd_new_feats:
                judge["queue"].enqueue(new_feats.detach())
            metric_totals["loss"] += float(loss.detach())
            metric_totals["jit_loss"] += float(jit_loss.detach())
            metric_totals["cycle_loss"] += float(cycle_loss.detach())
            metric_totals["kl_loss"] += float(kl_loss.detach())
            metric_totals["fd_loss"] += float(fd_loss.detach())
            metric_totals["vae_frac"] += float(vae_mask.mean().detach())
            metric_totals["start_std"] += float(vae_start.std().detach())
            metric_totals["mu_std"] += float(mu.std().detach())
            metric_totals["logvar_mean"] += float(logvar.mean().detach())
            for key, value in fd_metrics.items():
                fd_metric_totals[key] = fd_metric_totals.get(key, 0.0) + value

        scaler.unscale_(optimizer)
        average_gradients(model)
        params_for_norm = list(model.parameters())
        if encoder is not None:
            average_gradients(encoder)
            params_for_norm += list(encoder.parameters())
        grad_norm = torch.nn.utils.clip_grad_norm_(
            params_for_norm,
            args.grad_clip,
        ) if args.grad_clip > 0 else get_grad_norm(params_for_norm)
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
        sps = args.batch_size * grad_accum_steps / step_time if step_time > 0 else 0.0
        mem_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)
        metrics = {k: v / grad_accum_steps for k, v in metric_totals.items()}
        fd_metrics = {k: v / grad_accum_steps for k, v in fd_metric_totals.items()}
        fd_metrics_reduced = {
            k: reduce_float(torch.tensor(v, device="cuda"))
            for k, v in fd_metrics.items()
        }
        metric_logger.update(
            loss=reduce_float(torch.tensor(metrics["loss"], device="cuda")),
            jit_loss=reduce_float(torch.tensor(metrics["jit_loss"], device="cuda")),
            cycle_loss=reduce_float(torch.tensor(metrics["cycle_loss"], device="cuda")),
            kl_loss=reduce_float(torch.tensor(metrics["kl_loss"], device="cuda")),
            fd_loss=reduce_float(torch.tensor(metrics["fd_loss"], device="cuda")),
            vae_frac=reduce_float(torch.tensor(metrics["vae_frac"], device="cuda")),
            start_std=reduce_float(torch.tensor(metrics["start_std"], device="cuda")),
            mu_std=reduce_float(torch.tensor(metrics["mu_std"], device="cuda")),
            logvar_mean=reduce_float(torch.tensor(metrics["logvar_mean"], device="cuda")),
            grad_norm=grad_norm,
            lr=optimizer.param_groups[0]["lr"],
            **fd_metrics_reduced,
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

        if (args.vae_start_vis_every > 0 and encoder is not None
                and (args.current_step % args.vae_start_vis_every == 0
                     or args.current_step == args.total_steps)):
            save_post_visualization(args, model, encoder, vis_x0, vis_labels,
                                    step=args.current_step)

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
    parser.add_argument("--vae_start_ablation_mode",
                        choices=["vae_start", "continued", "cutoff_only"],
                        default="vae_start",
                        help="vae_start keeps the paired start branch; continued and "
                             "cutoff_only train ordinary Gaussian JiT, with cutoff_only "
                             "expected to pass a reduced --train_t_max.")
    parser.add_argument("--vae_start_tc", default=0.30, type=float,
                        help="distance from the start endpoint where VAE starts are used")
    parser.add_argument("--vae_start_pre_steps", default=2000, type=int)
    parser.add_argument("--vae_start_encoder_type", choices=["conv", "dinov2_latent", "dinov2_sphere"],
                        default="conv")
    parser.add_argument("--vae_start_hidden", default=64, type=int)
    parser.add_argument("--vae_start_lr", default=2e-4, type=float)
    parser.add_argument("--vae_start_kl_weight", default=0.25, type=float)
    parser.add_argument("--vae_start_cycle_weight", default=1.0, type=float)
    parser.add_argument("--vae_start_logvar_min", default=-6.0, type=float)
    parser.add_argument("--vae_start_logvar_max", default=2.0, type=float)
    parser.add_argument("--vae_start_sample_mode", choices=["posterior", "mean_shift"],
                        default="posterior")
    parser.add_argument("--vae_start_mean_scale", default=1.0, type=float)
    parser.add_argument("--vae_start_fd_weight", default=0.0, type=float,
                        help="weight for an additional random-start FD loss branch")
    parser.add_argument("--vae_start_encoder_ckpt", default="", type=str,
                        help="optional checkpoint containing a vae_start_encoder state dict")
    parser.add_argument("--freeze_vae_start_encoder", action="store_true",
                        help="freeze a loaded or pretrained VAE-start encoder during JiT post-training")
    parser.add_argument("--vae_start_vis_every", default=0, type=int,
                        help="save VAE-start/random-start post-training contact sheets every N steps")
    parser.add_argument("--vae_start_vis_images", default=16, type=int,
                        help="number of fixed real images in VAE-start post-training contact sheets")
    parser.add_argument("--dinov2_start_model", default="vit_base_patch14_dinov2.lvd142m",
                        type=str)
    parser.add_argument("--dinov2_start_patch_size", default=14, type=int,
                        help="DINOv2 backbone patch size; the start bridge grid is set by latent token count")
    parser.add_argument("--dinov2_start_latent_tokens", default=256, type=int)
    parser.add_argument("--dinov2_start_token_dim", default=64, type=int)
    parser.add_argument("--dinov2_start_train_backbone", action="store_true",
                        help="fine-tune the DINOv2 backbone instead of freezing it")
    parser.add_argument("--dinov2_start_freeze_decoder_backbone", action="store_true",
                        help="freeze the DINOv2 decoder backbone in dinov2_sphere mode")
    parser.add_argument("--dinov2_start_no_pretrained", action="store_true",
                        help="initialize the DINOv2 start encoder from scratch")
    parser.add_argument("--dinov2_start_pretrained_path", default="", type=str,
                        help="optional local DINO-family checkpoint to load instead of downloading pretrained weights")
    parser.add_argument("--dinov2_start_noise_angle", default=85.0, type=float,
                        help="max angular noise used by dinov2_sphere latent perturbation")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    sys.exit(train_post(args))
