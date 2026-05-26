#!/usr/bin/env python3
"""Visualize JiT-B deterministic DDIM/flow inversion startpoints.

The JiT denoiser in this repository uses the standard convention t=0 for data
and t=1 for the start/noise endpoint.  This script integrates the learned
velocity field from x0 at t=0 to t=1, then integrates the resulting startpoint
back to t=0 to check whether the inverted endpoint is actually useful.
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info
from torchvision.utils import make_grid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from main_fd import get_args_parser
from utils.builders import create_generation_model
from utils.checkpoint_util import ckpt_resume
from utils.data_util import center_crop_arr


class ImageListWithLabels(Dataset):
    def __init__(self, list_path: str | Path, image_root: str | Path, transform=None):
        self.image_root = Path(image_root)
        self.transform = transform
        self.items: list[tuple[Path, int]] = []
        with Path(list_path).open() as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                path = Path(parts[0])
                if not path.is_absolute():
                    path = self.image_root / path
                self.items.append((path, int(parts[1])))
        if not self.items:
            raise ValueError(f"no images listed in {list_path}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, label = self.items[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class HfParquetImageNetDataset(IterableDataset):
    def __init__(self, files: list[str], img_size: int):
        super().__init__()
        self.files = list(files)
        self.img_size = img_size
        self.to_tensor = transforms.ToTensor()

    def _files_for_worker(self):
        worker = get_worker_info()
        if worker is None:
            return self.files
        return self.files[worker.id::worker.num_workers]

    def __iter__(self):
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError("HF parquet ImageNet loading requires pyarrow") from exc

        for path in self._files_for_worker():
            table = pq.read_table(path)
            names = set(table.column_names)
            image_col = "image" if "image" in names else None
            label_col = "label" if "label" in names else ("labels" if "labels" in names else None)
            if image_col is None or label_col is None:
                raise ValueError(f"missing image/label columns in {path}: {table.column_names}")

            images = table[image_col].to_pylist()
            labels = table[label_col].to_pylist()
            for image_obj, label in zip(images, labels):
                if isinstance(image_obj, dict):
                    if image_obj.get("bytes") is not None:
                        image = Image.open(io.BytesIO(image_obj["bytes"]))
                    elif image_obj.get("path") is not None:
                        image = Image.open(image_obj["path"])
                    else:
                        raise ValueError(f"unsupported HF image object: {list(image_obj.keys())}")
                elif isinstance(image_obj, (bytes, bytearray)):
                    image = Image.open(io.BytesIO(image_obj))
                else:
                    raise TypeError(f"unsupported image object type: {type(image_obj)}")

                image = image.convert("RGB")
                image = center_crop_arr(image, self.img_size)
                yield self.to_tensor(image) * 2.0 - 1.0, int(label)


def build_loader(args):
    transform = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, args.img_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2.0 - 1.0),
    ])

    data_path = Path(args.data_path)
    if (data_path / "train").is_dir():
        dataset = datasets.ImageFolder(data_path / "train", transform=transform)
    elif args.image_list and args.image_root:
        dataset = ImageListWithLabels(args.image_list, args.image_root, transform=transform)
    else:
        parquet_files = sorted(glob.glob(str(data_path / "data" / "train-*.parquet")))
        if not parquet_files:
            raise FileNotFoundError(
                f"Could not find ImageFolder train/ or HF parquet shards under {data_path}"
            )
        dataset = HfParquetImageNetDataset(parquet_files, args.img_size)

    return DataLoader(
        dataset,
        batch_size=args.num_images,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )


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


def _velocity(model, z: torch.Tensor, t: torch.Tensor, labels: torch.Tensor, cfg: float):
    while t.ndim < z.ndim:
        t = t[..., None]
    return model._forward_with_cfg(z, t, labels, cfg=cfg, cfg_interval=None)


@torch.inference_mode()
def euler_integrate(model, z: torch.Tensor, labels: torch.Tensor,
                    *, t_start: float, t_end: float, num_steps: int, cfg: float):
    bsz = z.shape[0]
    ts = torch.linspace(t_start, t_end, num_steps + 1, device=z.device)
    for i in range(num_steps):
        t_cur = ts[i].expand(bsz)
        dt = (ts[i + 1] - ts[i]).view(1, 1, 1, 1)
        z = z + dt * _velocity(model, z, t_cur, labels, cfg)
    return z


def cosine_to_x0(z: torch.Tensor, x0: torch.Tensor):
    zf = z.detach().float().flatten(1)
    xf = x0.detach().float().flatten(1)
    return torch.nn.functional.cosine_similarity(zf, xf, dim=1).mean()


def main():
    parser = argparse.ArgumentParser(
        "Visualize JiT-B DDIM inversion",
        parents=[get_args_parser()],
        conflict_handler="resolve",
    )
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--num_images", default=16, type=int)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--inversion_steps", default=32, type=int)
    parser.add_argument("--reconstruction_steps", default=32, type=int)
    parser.add_argument("--inversion_cfg", default=1.0, type=float)
    parser.add_argument("--reconstruction_cfg", default=1.0, type=float)
    parser.add_argument("--seed", default=123, type=int)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    args.disable_vis = True
    args.enable_wandb = False
    args.output_dir = str(args.out_dir)
    args.project = "ddim_inversion_vis"
    args.exp_name = "vis"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model, ema_model = create_generation_model(args)
    ckpt_resume(args, model, optimizer=None, model_ema=ema_model)
    model.eval().cuda().requires_grad_(False)

    loader = build_loader(args)
    x0, labels = next(iter(loader))
    x0 = x0.cuda(non_blocking=True)[: args.num_images]
    labels = labels.cuda(non_blocking=True)[: args.num_images]

    eps_ref = torch.randn_like(x0) * args.noise_scale
    z_inv = euler_integrate(
        model, x0.clone(), labels,
        t_start=0.0, t_end=1.0,
        num_steps=args.inversion_steps, cfg=args.inversion_cfg,
    )
    recon = euler_integrate(
        model, z_inv.clone(), labels,
        t_start=1.0, t_end=0.0,
        num_steps=args.reconstruction_steps, cfg=args.reconstruction_cfg,
    )
    one_step = euler_integrate(
        model, z_inv.clone(), labels,
        t_start=1.0, t_end=0.0,
        num_steps=1, cfg=args.reconstruction_cfg,
    )
    random_one_step = euler_integrate(
        model, eps_ref.clone(), labels,
        t_start=1.0, t_end=0.0,
        num_steps=1, cfg=args.reconstruction_cfg,
    )
    random_multi = euler_integrate(
        model, eps_ref.clone(), labels,
        t_start=1.0, t_end=0.0,
        num_steps=args.reconstruction_steps, cfg=args.reconstruction_cfg,
    )

    stats = {
        "load_from": args.load_from,
        "num_images": int(x0.shape[0]),
        "inversion_steps": int(args.inversion_steps),
        "reconstruction_steps": int(args.reconstruction_steps),
        "inversion_cfg": float(args.inversion_cfg),
        "reconstruction_cfg": float(args.reconstruction_cfg),
        "z_inv_mean": float(z_inv.mean().item()),
        "z_inv_std": float(z_inv.std().item()),
        "z_inv_min": float(z_inv.min().item()),
        "z_inv_max": float(z_inv.max().item()),
        "eps_ref_std": float(eps_ref.std().item()),
        "z_inv_x0_cosine": float(cosine_to_x0(z_inv, x0).item()),
        "z_inv_eps_cosine": float(cosine_to_x0(z_inv, eps_ref).item()),
        "recon_mse": float(torch.mean((recon - x0) ** 2).item()),
        "one_step_mse": float(torch.mean((one_step - x0) ** 2).item()),
        "random_one_step_mse": float(torch.mean((random_one_step - x0) ** 2).item()),
        "random_multi_mse": float(torch.mean((random_multi - x0) ** 2).item()),
    }

    nrow = min(8, args.num_images)
    rows = [
        _row("real x0", _to_image_range(x0), nrow),
        _row("DDIM inversion z_inv", _to_noise_range(z_inv), nrow),
        _row("z_inv - eps_ref", _to_noise_range(z_inv - eps_ref), nrow),
        _row("recon from z_inv\nmulti-step", _to_image_range(recon), nrow),
        _row("recon from z_inv\none-step", _to_image_range(one_step), nrow),
        _row("random eps\none-step", _to_image_range(random_one_step), nrow),
        _row("random eps\nmulti-step", _to_image_range(random_multi), nrow),
    ]
    _save_contact(rows, args.out_dir / "ddim_inversion_contact.png")
    (args.out_dir / "ddim_inversion_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"saved {args.out_dir / 'ddim_inversion_contact.png'}")


if __name__ == "__main__":
    main()
