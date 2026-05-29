"""DINOv2 latent-token start encoder for JiT paired start experiments."""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import create_model
from timm.layers import trunc_normal_


def _rms_normalize(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    dims = tuple(range(1, z.ndim))
    rms = z.float().square().mean(dim=dims, keepdim=True).sqrt()
    return z / rms.clamp_min(eps).to(dtype=z.dtype)


def _remap_hf_dinov3_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if "embeddings.cls_token" not in state:
        return state

    remapped: dict[str, torch.Tensor] = {}
    direct = {
        "embeddings.cls_token": "cls_token",
        "embeddings.register_tokens": "reg_token",
        "embeddings.patch_embeddings.weight": "patch_embed.proj.weight",
        "embeddings.patch_embeddings.bias": "patch_embed.proj.bias",
        "norm.weight": "norm.weight",
        "norm.bias": "norm.bias",
    }
    for src, dst in direct.items():
        if src in state:
            remapped[dst] = state[src]

    layer_ids = sorted({
        int(key.split(".")[1])
        for key in state
        if key.startswith("layer.") and key.split(".")[1].isdigit()
    })
    for idx in layer_ids:
        prefix = f"layer.{idx}"
        block = f"blocks.{idx}"
        for src, dst in (
            ("norm1.weight", "norm1.weight"),
            ("norm1.bias", "norm1.bias"),
            ("norm2.weight", "norm2.weight"),
            ("norm2.bias", "norm2.bias"),
            ("attention.o_proj.weight", "attn.proj.weight"),
            ("attention.o_proj.bias", "attn.proj.bias"),
            ("mlp.up_proj.weight", "mlp.fc1.weight"),
            ("mlp.up_proj.bias", "mlp.fc1.bias"),
            ("mlp.down_proj.weight", "mlp.fc2.weight"),
            ("mlp.down_proj.bias", "mlp.fc2.bias"),
            ("layer_scale1.lambda1", "gamma_1"),
            ("layer_scale2.lambda1", "gamma_2"),
        ):
            key = f"{prefix}.{src}"
            if key in state:
                remapped[f"{block}.{dst}"] = state[key]

        qkv_w = [
            state.get(f"{prefix}.attention.{name}_proj.weight")
            for name in ("q", "k", "v")
        ]
        if all(t is not None for t in qkv_w):
            remapped[f"{block}.attn.qkv.weight"] = torch.cat(qkv_w, dim=0)
        qkv_b = [
            state.get(f"{prefix}.attention.{name}_proj.bias")
            for name in ("q", "k", "v")
        ]
        if all(t is not None for t in qkv_b):
            remapped[f"{block}.attn.qkv.bias"] = torch.cat(qkv_b, dim=0)

    return remapped


def _load_local_pretrained(model: nn.Module, checkpoint_path: str) -> None:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"DINO checkpoint not found: {path}")

    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(path), device="cpu")
    else:
        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict):
            for key in ("model", "state_dict", "module"):
                if key in state and isinstance(state[key], dict):
                    state = state[key]
                    break

    state = {
        key.removeprefix("module."): value
        for key, value in state.items()
        if torch.is_tensor(value)
    }
    state = _remap_hf_dinov3_state(state)
    model_state = model.state_dict()
    filtered = {}
    skipped_shape = []
    for key, value in state.items():
        if key not in model_state:
            continue
        if model_state[key].shape != value.shape:
            skipped_shape.append(key)
            continue
        filtered[key] = value
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(
        "[DINO-start] loaded local checkpoint "
        f"{path} matched={len(filtered)} missing={len(missing)} "
        f"unexpected={len(unexpected)} skipped_shape={len(skipped_shape)}",
        flush=True,
    )


class TokenToImage(nn.Module):
    """Linear unpatchify head used by DINO-token autoencoders."""

    def __init__(
        self,
        *,
        img_size: int = 256,
        patch_size: int = 16,
        in_dim: int = 64,
        out_channels: int = 6,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Linear(in_dim, out_channels * patch_size * patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, num_tokens, _ = x.shape
        grid = int(math.sqrt(num_tokens))
        if grid * grid != num_tokens:
            raise ValueError(f"num_tokens must be square, got {num_tokens}")
        if num_tokens != self.num_patches:
            raise ValueError(f"expected {self.num_patches} tokens, got {num_tokens}")

        p = self.patch_size
        c = self.out_channels
        x = self.proj(x)
        x = x.reshape(bsz, grid, grid, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(bsz, c, grid * p, grid * p)


class DINOv2LatentToImageBridge(nn.Module):
    """DINOv2-decoder-style bridge from latent tokens to pixel-space starts."""

    def __init__(
        self,
        *,
        img_size: int = 256,
        patch_size: int = 16,
        model_name: str = "vit_base_patch14_dinov2.lvd142m",
        num_latent_tokens: int = 256,
        pretrained: bool = True,
        pretrained_path: str = "",
        freeze_backbone: bool = False,
        out_channels: int = 3,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.model = create_model(
            model_name,
            pretrained=pretrained and not pretrained_path,
            img_size=img_size,
            patch_size=patch_size,
            drop_path_rate=0.0,
        )
        if pretrained_path:
            _load_local_pretrained(self.model, pretrained_path)
        self.embed_dim = self.model.embed_dim
        self.num_img_tokens = self.model.patch_embed.num_patches
        self.num_prefix_tokens = self.model.num_prefix_tokens
        self.num_latent_tokens = num_latent_tokens

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.latent_pos_embed = nn.Parameter(
            torch.zeros(1, num_latent_tokens, self.embed_dim)
        )
        nn.init.normal_(self.mask_token, std=1e-6)
        trunc_normal_(self.latent_pos_embed, std=0.02)

        self.to_image = TokenToImage(
            img_size=img_size,
            patch_size=patch_size,
            in_dim=self.embed_dim,
            out_channels=out_channels,
        )

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        # The decoder never consumes real image patches; keep patch embedding
        # frozen to avoid wasting optimizer state on an unused module.
        for param in self.model.patch_embed.parameters():
            param.requires_grad = False

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 3:
            raise ValueError(f"expected latent tokens [B,N,D], got {tuple(z.shape)}")
        if z.shape[1] != self.num_latent_tokens:
            raise ValueError(f"expected {self.num_latent_tokens} tokens, got {z.shape[1]}")
        if z.shape[2] != self.embed_dim:
            raise ValueError(f"expected token dim {self.embed_dim}, got {z.shape[2]}")

        x = self.mask_token.expand(z.shape[0], self.num_img_tokens, -1)
        with torch.cuda.amp.autocast(enabled=False):
            x = self.model._pos_embed(x)
            z = z + self.latent_pos_embed
            x = self.model.patch_drop(x)
            x = torch.cat([x, z], dim=1)

        temp = x.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype
        x = x.to(main_type)

        x = self.model.norm_pre(x)
        x = self.model.blocks(x)
        x = self.model.norm(x)
        x = x[:, self.num_prefix_tokens:self.num_prefix_tokens + self.num_img_tokens]
        out = self.to_image(x)
        if out.shape[-1] != self.img_size or out.shape[-2] != self.img_size:
            out = F.interpolate(
                out,
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
            )
        return out


class DINOv2LatentTokenizer(nn.Module):
    """DINOv2 encoder with appended latent query tokens.

    This mirrors the DINOv2 tokenizer pattern used in RobusTok: image patch
    tokens and learned latent tokens are passed through the pretrained DINOv2
    transformer together, and only the final latent tokens are returned.
    """

    SUPPORTED_MODELS = {
        "vit_small_patch14_dinov2.lvd142m",
        "vit_base_patch14_dinov2.lvd142m",
        "vit_large_patch14_dinov2.lvd142m",
        "vit_giant_patch14_dinov2.lvd142m",
        "vit_small_patch14_reg4_dinov2.lvd142m",
        "vit_base_patch14_reg4_dinov2.lvd142m",
        "vit_large_patch14_reg4_dinov2.lvd142m",
        "vit_giant_patch14_reg4_dinov2.lvd142m",
        "vit_small_patch16_dinov3",
        "vit_base_patch16_dinov3",
        "vit_large_patch16_dinov3",
        "vit_small_patch16_dinov3_qkvb",
        "vit_base_patch16_dinov3_qkvb",
        "vit_large_patch16_dinov3_qkvb",
    }

    def __init__(
        self,
        *,
        model_name: str = "vit_base_patch14_dinov2.lvd142m",
        img_size: int = 256,
        patch_size: int = 16,
        num_latent_tokens: int = 256,
        pretrained: bool = True,
        pretrained_path: str = "",
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"unsupported DINOv2 model: {model_name}")

        self.model = create_model(
            model_name,
            pretrained=pretrained and not pretrained_path,
            img_size=img_size,
            patch_size=patch_size,
            drop_path_rate=0.0,
        )
        if pretrained_path:
            _load_local_pretrained(self.model, pretrained_path)
        self.embed_dim = self.model.embed_dim
        self.num_img_tokens = self.model.patch_embed.num_patches
        self.num_prefix_tokens = self.model.num_prefix_tokens
        self.num_latent_tokens = num_latent_tokens

        self.latent_tokens = nn.Parameter(
            torch.zeros(1, num_latent_tokens, self.embed_dim)
        )
        self.latent_pos_embed = nn.Parameter(
            torch.zeros(1, num_latent_tokens, self.embed_dim)
        )
        nn.init.normal_(self.latent_tokens, std=1e-6)
        trunc_normal_(self.latent_pos_embed, std=0.02)

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model.patch_embed(x)
        with torch.cuda.amp.autocast(enabled=False):
            x = self.model._pos_embed(x)
            x = self.model.patch_drop(x)
            z = self.latent_tokens.expand(x.size(0), -1, -1)
            x = torch.cat([x, z + self.latent_pos_embed], dim=1)

        # Match timm's internal matmul dtype under autocast.
        temp = x.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype
        x = x.to(main_type)

        x = self.model.norm_pre(x)
        x = self.model.blocks(x)
        x = self.model.norm(x)
        return x[:, -self.num_latent_tokens:]


class DINOv2LatentStartEncoder(nn.Module):
    """Image-conditioned Gaussian start distribution from DINOv2 latent tokens."""

    def __init__(
        self,
        *,
        channels: int = 3,
        img_size: int = 256,
        patch_size: int = 16,
        model_name: str = "vit_base_patch14_dinov2.lvd142m",
        num_latent_tokens: int = 256,
        token_dim: int = 64,
        pretrained: bool = True,
        pretrained_path: str = "",
        freeze_backbone: bool = True,
        logvar_min: float = -6.0,
        logvar_max: float = 2.0,
    ) -> None:
        super().__init__()
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max

        self.register_buffer(
            "image_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

        self.tokenizer = DINOv2LatentTokenizer(
            model_name=model_name,
            img_size=img_size,
            patch_size=patch_size,
            num_latent_tokens=num_latent_tokens,
            pretrained=pretrained,
            pretrained_path=pretrained_path,
            freeze_backbone=freeze_backbone,
        )
        self.to_latent = nn.Sequential(
            nn.LayerNorm(self.tokenizer.embed_dim),
            nn.Linear(self.tokenizer.embed_dim, token_dim),
            nn.SiLU(),
            nn.Linear(token_dim, token_dim),
        )
        start_grid = int(math.sqrt(num_latent_tokens))
        if start_grid * start_grid != num_latent_tokens:
            raise ValueError(
                f"num_latent_tokens must form a square start grid, got {num_latent_tokens}"
            )
        if img_size % start_grid != 0:
            raise ValueError(
                f"img_size={img_size} must be divisible by start grid={start_grid}"
            )
        self.to_image = TokenToImage(
            img_size=img_size,
            patch_size=img_size // start_grid,
            in_dim=token_dim,
            out_channels=channels * 2,
        )

    def _normalize_for_dino(self, x0: torch.Tensor) -> torch.Tensor:
        x01 = (x0.float().clamp(-1, 1) + 1.0) * 0.5
        return ((x01 - self.image_mean) / self.image_std).to(dtype=x0.dtype)

    def latent_tokens(self, x0: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(self._normalize_for_dino(x0))
        return _rms_normalize(self.to_latent(tokens))

    def stats(self, x0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.latent_tokens(x0)
        mu, logvar = self.to_image(h).chunk(2, dim=1)
        logvar = logvar.clamp(self.logvar_min, self.logvar_max)
        return mu, logvar

    def forward(self, x0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.stats(x0)
        eps = torch.randn_like(mu)
        return mu + torch.exp(0.5 * logvar) * eps, mu, logvar


class DINOv2SphereStartEncoder(nn.Module):
    """Sphere-latent start bridge with DINOv2 tokens kept at [B, 256, 768]."""

    start_kind = "sphere_latent"

    def __init__(
        self,
        *,
        channels: int = 3,
        img_size: int = 256,
        patch_size: int = 16,
        model_name: str = "vit_base_patch14_dinov2.lvd142m",
        num_latent_tokens: int = 256,
        pretrained: bool = True,
        pretrained_path: str = "",
        freeze_encoder_backbone: bool = True,
        freeze_decoder_backbone: bool = False,
        noise_sigma_max_angle: float = 85.0,
    ) -> None:
        super().__init__()
        self.noise_sigma_max_angle = noise_sigma_max_angle
        self.tokenizer = DINOv2LatentTokenizer(
            model_name=model_name,
            img_size=img_size,
            patch_size=patch_size,
            num_latent_tokens=num_latent_tokens,
            pretrained=pretrained,
            pretrained_path=pretrained_path,
            freeze_backbone=freeze_encoder_backbone,
        )
        self.bridge = DINOv2LatentToImageBridge(
            img_size=img_size,
            patch_size=patch_size,
            model_name=model_name,
            num_latent_tokens=num_latent_tokens,
            pretrained=pretrained,
            pretrained_path=pretrained_path,
            freeze_backbone=freeze_decoder_backbone,
            out_channels=channels,
        )
        if self.tokenizer.embed_dim != self.bridge.embed_dim:
            raise ValueError(
                f"encoder dim {self.tokenizer.embed_dim} != decoder dim {self.bridge.embed_dim}"
            )

        self.register_buffer(
            "image_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    def _normalize_for_dino(self, x0: torch.Tensor) -> torch.Tensor:
        x01 = (x0.float().clamp(-1, 1) + 1.0) * 0.5
        return ((x01 - self.image_mean) / self.image_std).to(dtype=x0.dtype)

    def latent_tokens(self, x0: torch.Tensor) -> torch.Tensor:
        return _rms_normalize(self.tokenizer(self._normalize_for_dino(x0)))

    def spherify(self, z: torch.Tensor) -> torch.Tensor:
        return _rms_normalize(z)

    def sample_latent(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        eps = torch.randn_like(z)
        sigma = math.tan(math.radians(self.noise_sigma_max_angle))
        radius_shape = (z.shape[0],) + (1,) * (z.ndim - 1)
        radius = torch.rand(radius_shape, device=z.device, dtype=z.dtype)
        z_noisy = self.spherify(z + radius * sigma * eps)
        return z_noisy, eps, radius

    def sample_start(self, x0: torch.Tensor) -> dict[str, torch.Tensor]:
        z_clean = self.latent_tokens(x0)
        z_noisy, eps, radius = self.sample_latent(z_clean)
        start = self.bridge(z_noisy)
        clean_start = self.bridge(z_clean)
        random_latent = self.spherify(torch.randn_like(z_clean))
        random_start = self.bridge(random_latent)
        latent_cosine = F.cosine_similarity(
            z_clean.flatten(1).float(),
            z_noisy.flatten(1).float(),
            dim=1,
        ).mean()
        return {
            "start": start,
            "clean_start": clean_start,
            "random_start": random_start,
            "latent_clean": z_clean,
            "latent_noisy": z_noisy,
            "latent_eps": eps,
            "latent_radius": radius,
            "latent_cosine": latent_cosine,
        }

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        return self.sample_start(x0)["start"]
