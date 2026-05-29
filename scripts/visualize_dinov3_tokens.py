"""Visualize DINO-family patch and sphere latent tokens with PCA and norm maps."""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from timm.models import create_model
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main_jit_vae_start import HfParquetImageNetDataset
from models.dinov2_start import DINOv2SphereStartEncoder, _load_local_pretrained


def to_image_range(x: torch.Tensor) -> torch.Tensor:
    return (x.detach().float().clamp(-1, 1) + 1.0) * 0.5


def minmax_image(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().float()
    flat = x.flatten(1)
    lo = flat.min(dim=1).values.view(-1, 1, 1, 1)
    hi = flat.max(dim=1).values.view(-1, 1, 1, 1)
    return ((x - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1)


def tokens_pca_rgb(tokens: torch.Tensor, image_size: int) -> torch.Tensor:
    tokens = tokens.detach().float().cpu()
    grid = int(tokens.shape[1] ** 0.5)
    if grid * grid != tokens.shape[1]:
        raise ValueError(f"token count must be square, got {tokens.shape[1]}")
    images = []
    for one in tokens:
        x = one - one.mean(dim=0, keepdim=True)
        _, _, v = torch.pca_lowrank(x, q=3, center=False)
        rgb = x @ v[:, :3]
        for channel in range(3):
            if rgb[:, channel].abs().max() and rgb[:, channel].mean() < 0:
                rgb[:, channel] = -rgb[:, channel]
        rgb = rgb.T.reshape(1, 3, grid, grid)
        rgb = torch.nn.functional.interpolate(rgb, size=(image_size, image_size), mode="nearest")
        images.append(minmax_image(rgb)[0])
    return torch.stack(images, dim=0)


def tokens_norm_heatmap(tokens: torch.Tensor, image_size: int) -> torch.Tensor:
    tokens = tokens.detach().float()
    grid = int(tokens.shape[1] ** 0.5)
    norm = tokens.norm(dim=-1).reshape(tokens.shape[0], 1, grid, grid)
    norm = torch.nn.functional.interpolate(norm, size=(image_size, image_size), mode="nearest")
    v = minmax_image(norm)
    red = v
    green = 1.0 - (2.0 * v - 1.0).abs()
    blue = 1.0 - v
    return torch.cat([red, green.clamp(0, 1), blue], dim=1)


def row(label: str, images: torch.Tensor, nrow: int, label_width: int = 360) -> Image.Image:
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


def save_contact(rows: list[Image.Image], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = max(r.width for r in rows)
    height = sum(r.height for r in rows)
    canvas = Image.new("RGB", (width, height), "white")
    y = 0
    for one in rows:
        canvas.paste(one, (0, y))
        y += one.height
    canvas.save(path)


def normalize_for_dino(x0: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=x0.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x0.device).view(1, 3, 1, 1)
    x01 = (x0.float().clamp(-1, 1) + 1.0) * 0.5
    return (x01 - mean) / std


def load_images(data_path: str, image_size: int, num_images: int) -> tuple[torch.Tensor, torch.Tensor]:
    files = sorted(glob.glob(str(Path(data_path) / "data" / "train-*.parquet")))
    if not files:
        raise FileNotFoundError(f"no HF parquet shards under {data_path}/data")
    loader = DataLoader(
        HfParquetImageNetDataset(files[:1], image_size),
        batch_size=num_images,
        num_workers=0,
    )
    return next(iter(loader))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--encoder_ckpt", default="")
    parser.add_argument("--image_size", default=256, type=int)
    parser.add_argument("--num_images", default=4, type=int)
    parser.add_argument("--dino_model", default="vit_base_patch16_dinov3")
    parser.add_argument("--dino_patch_size", default=16, type=int)
    parser.add_argument("--dino_pretrained_path", required=True)
    parser.add_argument("--noise_angle", default=85.0, type=float)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    x0, _ = load_images(args.data_path, args.image_size, args.num_images)
    x0 = x0.to(device)
    nrow = min(args.num_images, 4)

    dino = create_model(
        args.dino_model,
        pretrained=False,
        img_size=args.image_size,
        patch_size=args.dino_patch_size,
    ).to(device)
    _load_local_pretrained(dino, args.dino_pretrained_path)
    dino.eval()
    with torch.no_grad():
        features = dino.forward_features(normalize_for_dino(x0))
        patch_tokens = features[:, dino.num_prefix_tokens:]

    rows = [
        row("real x0", to_image_range(x0), nrow),
        row("DINOv3 patch tokens\nPCA RGB", tokens_pca_rgb(patch_tokens, args.image_size), nrow),
        row("DINOv3 patch tokens\ntoken norm heatmap", tokens_norm_heatmap(patch_tokens, args.image_size), nrow),
    ]

    if args.encoder_ckpt:
        encoder = DINOv2SphereStartEncoder(
            img_size=args.image_size,
            patch_size=args.dino_patch_size,
            model_name=args.dino_model,
            num_latent_tokens=256,
            pretrained=True,
            pretrained_path=args.dino_pretrained_path,
            noise_sigma_max_angle=args.noise_angle,
        ).to(device)
        payload = torch.load(args.encoder_ckpt, map_location="cpu")
        encoder.load_state_dict(payload["vae_start_encoder"], strict=True)
        encoder.eval()
        with torch.no_grad():
            sample = encoder.sample_start(x0)
        rows.extend([
            row("sphere latent clean\nPCA RGB", tokens_pca_rgb(sample["latent_clean"], args.image_size), nrow),
            row("sphere latent clean\ntoken norm heatmap", tokens_norm_heatmap(sample["latent_clean"], args.image_size), nrow),
            row("sphere latent noisy\nPCA RGB", tokens_pca_rgb(sample["latent_noisy"], args.image_size), nrow),
            row("sphere latent noisy\ntoken norm heatmap", tokens_norm_heatmap(sample["latent_noisy"], args.image_size), nrow),
            row("clean bridge start", minmax_image(sample["clean_start"]), nrow),
            row("sampled bridge start", minmax_image(sample["start"]), nrow),
            row("random sphere bridge start", minmax_image(sample["random_start"]), nrow),
        ])

    save_contact(rows, out_dir / "dinov3_token_contact.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
