"""Utilities for sampling and normalizing generation start points."""

from __future__ import annotations

import torch


START_SUPPORT_MODES = ("gaussian", "pixel_sphere")


def rms_normalize_to_noise_scale(
    z: torch.Tensor,
    noise_scale: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Project each sample to fixed RMS, matching Gaussian noise scale."""
    dims = tuple(range(1, z.ndim))
    rms = z.float().square().mean(dim=dims, keepdim=True).sqrt()
    return z * (float(noise_scale) / rms.clamp_min(eps)).to(dtype=z.dtype)


def apply_start_support(
    z: torch.Tensor,
    mode: str = "gaussian",
    noise_scale: float = 1.0,
) -> torch.Tensor:
    """Map a raw start tensor onto the requested support."""
    if mode == "gaussian":
        return z
    if mode == "pixel_sphere":
        return rms_normalize_to_noise_scale(z, noise_scale=noise_scale)
    raise ValueError(f"unknown start support mode: {mode}")


def sample_start(
    shape: tuple[int, ...],
    *,
    device: torch.device | str,
    dtype: torch.dtype | None = None,
    noise_scale: float = 1.0,
    mode: str = "gaussian",
    same_noise: bool = False,
) -> torch.Tensor:
    """Sample a start tensor from the configured inference support."""
    if len(shape) < 2:
        raise ValueError(f"expected NCHW-like shape, got {shape}")

    base_shape = (1, *shape[1:]) if same_noise else shape
    z = torch.randn(base_shape, device=device, dtype=dtype)
    if mode == "gaussian":
        z = z * float(noise_scale)
    elif mode == "pixel_sphere":
        z = rms_normalize_to_noise_scale(z, noise_scale=noise_scale)
    else:
        raise ValueError(f"unknown start support mode: {mode}")

    if same_noise:
        z = z.repeat(shape[0], *([1] * (len(shape) - 1)))
    return z
