"""Compute pMF self-representation FD reference statistics.

The output is compatible with ``main_fd.py --fd_repr_models self_pmf_b``.
Images are loaded from ImageNet ``train/``, converted to model range
``[-1, 1]``, then passed through a frozen pMF-B judge at a low-noise point.
"""

import argparse
import logging
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from frechet_distance.self_repr import (
    build_pmf_b_model_from_args,
    build_pmf_self_feature_extractor,
    load_pmf_b_checkpoint,
    self_pmf_stats_name,
)
from utils.data_util import center_crop_arr
from utils.distributed_util import enable_distributed, get_global_rank, get_world_size


logger = logging.getLogger("FD_loss")


def parse_args():
    p = argparse.ArgumentParser(description="Compute pMF self-FD reference stats")
    p.add_argument("--data_path", type=str, default="data/imagenet",
                   help="ImageNet root dir with a train/ subfolder")
    p.add_argument("--load_from", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="data/fid_stats")
    p.add_argument("--output_name", type=str, default=None)
    p.add_argument("--num_images", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=10)
    p.add_argument("--seed", type=int, default=1)

    p.add_argument("--model", default="pMF_B", type=str)
    p.add_argument("--img_size", default=256, type=int)
    p.add_argument("--patch_size", default=16, type=int)
    p.add_argument("--num_classes", default=1000, type=int)
    p.add_argument("--token_channels", default=3, type=int)
    p.add_argument("--tokenizer_patch_size", default=1, type=int)
    p.add_argument("--label_drop_prob", default=0.1, type=float)
    p.add_argument("--P_mean", type=float, default=0.8)
    p.add_argument("--P_std", type=float, default=0.8)
    p.add_argument("--ratio_r_neq_t", type=float, default=0.5)
    p.add_argument("--cfg_beta", type=float, default=1.0)
    p.add_argument("--cfg_omega_max", type=float, default=7.0)
    p.add_argument("--aux_head_depth", type=int, default=8)
    p.add_argument("--class_tokens", type=int, default=8)
    p.add_argument("--time_tokens", type=int, default=4)
    p.add_argument("--guidance_tokens", type=int, default=4)
    p.add_argument("--interval_tokens", type=int, default=2)
    p.add_argument("--norm_eps", type=float, default=0.01)
    p.add_argument("--norm_p", type=float, default=1.0)
    p.add_argument("--t_eps", type=float, default=0.05)
    p.add_argument("--noise_scale", type=float, default=1.0)
    p.add_argument("--tr_uniform", action="store_true")
    p.add_argument("--rope_2d", action="store_true")
    p.add_argument("--learned_pe", action="store_true")
    p.add_argument("--disable_v_head", action="store_true")
    p.add_argument("--perceptual_threshold", type=float, default=0.8)
    p.add_argument("--perceptual_loss_on_aux", action="store_true")

    p.add_argument("--fd_self_shared_block", type=int, default=7)
    p.add_argument("--fd_self_t", type=float, default=0.05)
    p.add_argument("--cfg", type=float, default=8.5)
    p.add_argument("--interval_min", type=float, default=0.1)
    p.add_argument("--interval_max", type=float, default=0.7)
    return p.parse_args()


def setup_distributed(seed):
    enable_distributed()
    rank = get_global_rank()
    world_size = get_world_size()
    if torch.cuda.is_available():
        torch.cuda.set_device(rank % torch.cuda.device_count())
    torch.manual_seed(seed + rank)
    return rank, world_size


def build_dataloader(args, rank, world_size):
    transform = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, args.img_size)),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(os.path.join(args.data_path, "train"), transform=transform)
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False,
    ) if world_size > 1 else None
    loader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler, shuffle=False,
        drop_last=False, num_workers=args.num_workers, pin_memory=True,
    )
    return loader, len(dataset)


@torch.inference_mode()
def extract_stats(extractor, loader, feat_dim, rank, world_size, max_images_per_rank=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat_sum = torch.zeros(feat_dim, dtype=torch.float64, device=device)
    feat_outer = torch.zeros(feat_dim, feat_dim, dtype=torch.float64, device=device)
    count = 0

    pbar = tqdm(loader, desc=f"[rank {rank}] self-pMF features", position=rank, disable=rank != 0)
    for images, labels in pbar:
        if max_images_per_rank is not None and count >= max_images_per_rank:
            break
        if max_images_per_rank is not None:
            keep = min(images.shape[0], max_images_per_rank - count)
            images = images[:keep]
            labels = labels[:keep]

        images = images.to(device, non_blocking=True) * 2.0 - 1.0
        labels = labels.to(device, non_blocking=True)
        feats, _ = extractor(images, labels)
        feats64 = feats.double()
        feat_sum.add_(feats64.sum(0))
        feat_outer.addmm_(feats64.T, feats64)
        count += feats.shape[0]
        pbar.set_postfix({"images": count})

    if world_size > 1:
        dist.reduce(feat_sum, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(feat_outer, dst=0, op=dist.ReduceOp.SUM)
        count_t = torch.tensor([count], dtype=torch.long, device=device)
        dist.reduce(count_t, dst=0, op=dist.ReduceOp.SUM)
        count = int(count_t.item())

    if rank != 0:
        return None, None, count

    s_np = feat_sum.cpu().numpy()
    mu = s_np / count
    sigma = (feat_outer.cpu().numpy() - np.outer(s_np, s_np) / count) / (count - 1)
    return mu, sigma, count


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    rank, world_size = setup_distributed(args.seed)
    if rank != 0:
        logger.setLevel(logging.WARNING)

    logger.info(
        f"Computing self-pMF stats: model={args.model}, block={args.fd_self_shared_block}, "
        f"t={args.fd_self_t}, gpus={world_size}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_pmf_b_model_from_args(args, device=device)
    load_pmf_b_checkpoint(model, args.load_from)
    extractor = build_pmf_self_feature_extractor(
        model,
        shared_block_idx=args.fd_self_shared_block,
        t_self=args.fd_self_t,
        cfg=args.cfg,
        interval_min=args.interval_min,
        interval_max=args.interval_max,
    ).to(device).eval()

    loader, total_images = build_dataloader(args, rank, world_size)
    if args.num_images is not None:
        total_images = min(total_images, args.num_images)
    max_per_rank = (total_images + world_size - 1) // world_size
    logger.info(f"Dataset images used: {total_images} ({max_per_rank} per rank)")

    t0 = time.perf_counter()
    mu, sigma, count = extract_stats(
        extractor, loader, extractor.feat_dim, rank, world_size,
        max_images_per_rank=max_per_rank,
    )
    elapsed = time.perf_counter() - t0
    logger.info(f"Processed {count} images in {elapsed:.1f}s")

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        output_name = args.output_name or self_pmf_stats_name(
            args.fd_self_shared_block, args.fd_self_t, args.img_size,
        )
        out_path = os.path.join(args.output_dir, output_name)
        np.savez(out_path, mu=mu, sigma=sigma)
        logger.info(f"Saved {out_path} (n={count}, feat_dim={mu.shape[0]})")

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
