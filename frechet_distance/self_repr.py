"""Self-representation feature extractors for FD training.

These judges reuse a frozen diffusion/flow model as the feature extractor.
The first implementation targets pMF-B and extracts pooled patch tokens from
the shared MiT trunk at a low-noise point.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

import models
from models.denoiser_pmf import convert_pmf_checkpoint


logger = logging.getLogger("FD_loss")


class PmfSelfFeatureExtractor(nn.Module):
    """Frozen pMF internal-feature extractor.

    Input images are expected in model range ``[-1, 1]``. The extractor adds a
    small flow-matching noise level and returns mean-pooled patch features from
    a selected shared MiT block. Parameters are frozen, but gradients still flow
    from the output features to the input images during FD training.
    """

    def __init__(
        self,
        denoiser: nn.Module,
        shared_block_idx: int = 7,
        t_self: float = 0.05,
        cfg: float = 8.5,
        interval_min: float = 0.1,
        interval_max: float = 0.7,
    ):
        super().__init__()
        self.denoiser = denoiser.eval().requires_grad_(False)
        self.net = self.denoiser.net
        self.shared_block_idx = int(shared_block_idx)
        self.t_self = float(t_self)
        self.cfg = float(cfg)
        self.interval_min = float(interval_min)
        self.interval_max = float(interval_max)
        self.noise_scale = float(getattr(self.denoiser, "noise_scale", 1.0))
        self.feat_dim = int(self.net.hidden_size)

        if self.shared_block_idx < 0 or self.shared_block_idx >= len(self.net.shared_blocks):
            raise ValueError(
                f"shared_block_idx={self.shared_block_idx} is out of range for "
                f"{len(self.net.shared_blocks)} shared blocks"
            )

    def forward(self, images: torch.Tensor, labels: torch.Tensor):
        if labels is None:
            raise ValueError("PmfSelfFeatureExtractor requires class labels")

        bsz = images.shape[0]
        dtype = images.dtype
        device = images.device
        t = torch.full((bsz,), self.t_self, dtype=dtype, device=device)
        eps = torch.randn_like(images) * self.noise_scale
        z_t = (1.0 - self.t_self) * images + self.t_self * eps

        omega = torch.full((bsz,), self.cfg, dtype=dtype, device=device)
        t_min = torch.full((bsz,), self.interval_min, dtype=dtype, device=device)
        t_max = torch.full((bsz,), self.interval_max, dtype=dtype, device=device)

        seq = self.net._build_sequence(
            z_t, h=t, omega=omega, t_min=t_min, t_max=t_max, y=labels,
        )
        for idx, block in enumerate(self.net.shared_blocks):
            seq = block(seq, self.net.rope_freqs)
            if idx == self.shared_block_idx:
                break

        patch_tokens = seq[:, self.net.prefix_tokens:]
        return patch_tokens.mean(dim=1), None


def build_pmf_b_model_from_args(args, device: str | torch.device = "cuda") -> nn.Module:
    """Create a pMF model using the architecture flags needed by FD scripts."""
    if args.model not in models.pMFDenoiser_models:
        raise ValueError(f"self-FD pMF extractor requires a pMF model, got {args.model}")

    model = models.pMFDenoiser_models[args.model](
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_channels=getattr(args, "token_channels", 3),
        tokenizer_patch_size=getattr(args, "tokenizer_patch_size", 1),
        num_classes=args.num_classes,
        label_drop_prob=getattr(args, "label_drop_prob", 0.1),
        P_mean=getattr(args, "P_mean", 0.8),
        P_std=getattr(args, "P_std", 0.8),
        ratio_r_neq_t=getattr(args, "ratio_r_neq_t", 0.5),
        cfg_beta=getattr(args, "cfg_beta", 1.0),
        tr_uniform=getattr(args, "tr_uniform", False),
        cfg_omega_max=getattr(args, "cfg_omega_max", 7.0),
        aux_head_depth=getattr(args, "aux_head_depth", 8),
        class_tokens=getattr(args, "class_tokens", 8),
        time_tokens=getattr(args, "time_tokens", 4),
        guidance_tokens=getattr(args, "guidance_tokens", 4),
        interval_tokens=getattr(args, "interval_tokens", 2),
        t_eps=getattr(args, "t_eps", 0.05),
        perceptual_threshold=getattr(args, "perceptual_threshold", 0.8),
        perceptual_loss_on_aux=getattr(args, "perceptual_loss_on_aux", False),
        rope_2d=getattr(args, "rope_2d", False),
        learned_pe=getattr(args, "learned_pe", False),
        disable_v_head=getattr(args, "disable_v_head", False),
        noise_scale=getattr(args, "noise_scale", None),
        norm_eps=getattr(args, "norm_eps", 1e-4),
        norm_p=getattr(args, "norm_p", 1.0),
    )
    return model.to(device)


def load_pmf_b_checkpoint(model: nn.Module, checkpoint_path: str) -> nn.Module:
    """Load a pMF checkpoint into ``model`` using the repo's key conversion."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    state_dict = convert_pmf_checkpoint(state_dict)
    msg = model.load_state_dict(state_dict, strict=False)
    logger.info(f"[Self-FD] Loaded pMF checkpoint from {checkpoint_path}: {msg}")
    return model


def build_pmf_self_feature_extractor(
    denoiser: nn.Module,
    shared_block_idx: int = 7,
    t_self: float = 0.05,
    cfg: float = 8.5,
    interval_min: float = 0.1,
    interval_max: float = 0.7,
) -> PmfSelfFeatureExtractor:
    return PmfSelfFeatureExtractor(
        denoiser=denoiser,
        shared_block_idx=shared_block_idx,
        t_self=t_self,
        cfg=cfg,
        interval_min=interval_min,
        interval_max=interval_max,
    )


def self_pmf_stats_name(shared_block_idx: int = 7, t_self: float = 0.05, img_size: int = 256):
    t_code = int(round(float(t_self) * 100.0))
    return f"self_pmf_b_shared{int(shared_block_idx)}_t{t_code:03d}_in{int(img_size)}_stats.npz"
