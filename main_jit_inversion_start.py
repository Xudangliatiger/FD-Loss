"""JiT-B inversion-start paired post-training.

This entry point uses a frozen JiT teacher to compute deterministic inversion
startpoints z_inv(x0) online, then post-trains a student JiT to use those
paired startpoints near the t=1 start/noise endpoint.

Time convention in this repository:
    t = 0 is data, t = 1 is the start/noise endpoint.
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import sys
import time

import torch
import torch.nn.functional as F

import models
from main_fd import average_gradients, get_args_parser
from main_jit_vae_start import (
    build_paired_loader,
    infinite_loader,
    predict_x0,
    reduce_float,
    velocity_from_x0,
)
from utils.builders import create_generation_model
from utils.checkpoint_util import AsyncCheckpointSaver, ckpt_resume, save_checkpoint
from utils.distributed_util import (
    is_enabled,
    preempt_requested,
    register_preempt_handler,
)
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


def load_jit_teacher(args):
    if args.model not in models.JiTDenoiser_models:
        raise ValueError("inversion-start post-training currently supports JiT models only")
    teacher = models.JiTDenoiser_models[args.model](
        img_size=args.img_size,
        num_classes=args.num_classes,
        label_drop_prob=args.label_drop_prob,
        attn_dropout=args.attn_dropout,
        proj_dropout=args.proj_dropout,
        P_mean=args.P_mean,
        P_std=args.P_std,
        t_eps=args.t_eps,
        rope_2d=args.rope_2d,
        learned_pe=args.learned_pe,
        legacy_time_convention=args.legacy_time_convention,
    ).cuda()
    checkpoint = torch.load(args.inversion_teacher_ckpt, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    msg = teacher.load_state_dict(state_dict, strict=False)
    logger.info("[Inversion teacher] Loaded %s: %s", args.inversion_teacher_ckpt, msg)
    teacher.eval().requires_grad_(False)
    return teacher


def _velocity(model, z: torch.Tensor, t: torch.Tensor, labels: torch.Tensor, cfg: float):
    while t.ndim < z.ndim:
        t = t[..., None]
    return model._forward_with_cfg(z, t, labels, cfg=cfg, cfg_interval=None)


@torch.no_grad()
def invert_to_start(teacher, x0: torch.Tensor, labels: torch.Tensor,
                    *, num_steps: int, cfg: float):
    z = x0
    bsz = z.shape[0]
    ts = torch.linspace(0.0, 1.0, num_steps + 1, device=z.device)
    for i in range(num_steps):
        t_cur = ts[i].expand(bsz)
        dt = (ts[i + 1] - ts[i]).view(1, 1, 1, 1)
        z = z + dt * _velocity(teacher, z, t_cur, labels, cfg)
    return z


def train_post(args):
    setup(args)
    register_preempt_handler()

    model, ema_model = create_generation_model(args)
    ckpt_resume(args, model, optimizer=None, model_ema=ema_model)
    if args.model != "JiT_B":
        raise ValueError("main_jit_inversion_start.py currently supports --model JiT_B only")

    teacher = load_jit_teacher(args)
    loader, sampler = build_paired_loader(args)
    loader_iter = infinite_loader(loader, sampler)

    model.train().requires_grad_(True)
    optimizer = create_optimizer(args, model, print_trainable_params=True)
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
        "[Inversion-start] raw JiT t in [%.3f, %.3f], paired start where "
        "distance-from-start < %.3f (raw t >= %.3f), inversion_steps=%d, cfg=%.3f",
        args.train_t_min,
        args.train_t_max,
        args.inversion_start_tc,
        1.0 - args.inversion_start_tc,
        args.inversion_steps,
        args.inversion_cfg,
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
            "inversion_start_config": {
                "inversion_teacher_ckpt": args.inversion_teacher_ckpt,
                "inversion_start_tc": args.inversion_start_tc,
                "inversion_steps": args.inversion_steps,
                "inversion_cfg": args.inversion_cfg,
                "inversion_cycle_weight": args.inversion_cycle_weight,
                "train_t_min": args.train_t_min,
                "train_t_max": args.train_t_max,
            },
        }
        save_checkpoint(args, step, model, optimizer, ema_model, elapsed,
                        saver=saver, extra=extra)
        if is_enabled():
            torch.distributed.barrier()

    for step, _ in metric_logger.log_every(
        iter(int, 1), args.print_freq, header="Inversion-start:",
        start_iteration=args.current_step, n_iterations=args.total_steps,
    ):
        adjust_learning_rate(optimizer, step, args)
        optimizer.zero_grad(set_to_none=True)
        metric_totals = {
            "loss": 0.0,
            "jit_loss": 0.0,
            "cycle_loss": 0.0,
            "inv_frac": 0.0,
            "z_inv_std": 0.0,
            "z_inv_mean": 0.0,
            "eps_std": 0.0,
        }
        for _ in range(grad_accum_steps):
            x0, labels = next(loader_iter)
            x0 = x0.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            t = args.train_t_min + (args.train_t_max - args.train_t_min) * torch.rand(
                x0.shape[0], device=x0.device,
            )
            t_view = t.view(-1, 1, 1, 1)
            eps = torch.randn_like(x0) * args.noise_scale

            with torch.amp.autocast("cuda", enabled=args.enable_amp, dtype=args.amp_dtype):
                z_inv = invert_to_start(
                    teacher, x0, labels,
                    num_steps=args.inversion_steps,
                    cfg=args.inversion_cfg,
                )
                inv_mask = (t >= (1.0 - args.inversion_start_tc)).float().view(-1, 1, 1, 1)
                start = inv_mask * z_inv + (1.0 - inv_mask) * eps
                x_t = (1.0 - t_view) * x0 + t_view * start

                x0_pred = predict_x0(model, x_t, t, labels, drop_labels=True)
                v_pred = velocity_from_x0(x0_pred, x_t, t, args.t_eps)
                v_true = velocity_from_x0(x0, x_t, t, args.t_eps)
                jit_loss = F.mse_loss(v_pred, v_true)

                ones = torch.ones_like(t)
                cycle_pred = predict_x0(model, z_inv, ones, labels, drop_labels=False)
                cycle_loss = F.mse_loss(cycle_pred, x0)
                loss = jit_loss + args.inversion_cycle_weight * cycle_loss

            scaler.scale(loss / grad_accum_steps).backward()
            metric_totals["loss"] += float(loss.detach())
            metric_totals["jit_loss"] += float(jit_loss.detach())
            metric_totals["cycle_loss"] += float(cycle_loss.detach())
            metric_totals["inv_frac"] += float(inv_mask.mean().detach())
            metric_totals["z_inv_std"] += float(z_inv.std().detach())
            metric_totals["z_inv_mean"] += float(z_inv.mean().detach())
            metric_totals["eps_std"] += float(eps.std().detach())

        scaler.unscale_(optimizer)
        average_gradients(model)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            args.grad_clip,
        ) if args.grad_clip > 0 else get_grad_norm(model.parameters())
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
        metric_logger.update(
            loss=reduce_float(torch.tensor(metrics["loss"], device="cuda")),
            jit_loss=reduce_float(torch.tensor(metrics["jit_loss"], device="cuda")),
            cycle_loss=reduce_float(torch.tensor(metrics["cycle_loss"], device="cuda")),
            inv_frac=reduce_float(torch.tensor(metrics["inv_frac"], device="cuda")),
            z_inv_std=reduce_float(torch.tensor(metrics["z_inv_std"], device="cuda")),
            z_inv_mean=reduce_float(torch.tensor(metrics["z_inv_mean"], device="cuda")),
            eps_std=reduce_float(torch.tensor(metrics["eps_std"], device="cuda")),
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
        "JiT inversion-start paired post-training", parents=[get_args_parser()],
    )
    parser.add_argument("--image_root", default="", type=str,
                        help="root for paths in --image_list, usually ImageNet/train")
    parser.add_argument("--image_list", default="", type=str,
                        help="optional list file with 'relative_path label' rows")
    parser.add_argument("--train_t_min", default=0.02, type=float)
    parser.add_argument("--train_t_max", default=1.0, type=float)
    parser.add_argument("--inversion_teacher_ckpt", required=True, type=str)
    parser.add_argument("--inversion_start_tc", default=0.30, type=float,
                        help="distance from the start endpoint where z_inv starts are used")
    parser.add_argument("--inversion_steps", default=16, type=int)
    parser.add_argument("--inversion_cfg", default=1.0, type=float)
    parser.add_argument("--inversion_cycle_weight", default=1.0, type=float)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    sys.exit(train_post(args))
