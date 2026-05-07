#!/usr/bin/env python3
import argparse
import glob
import os
import random
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class DatasetMeta:
    grid_size: int
    dx: float


class Tdse2dPairDataset(Dataset):
    def __init__(self, sample_paths: List[str]):
        self.sample_paths = sample_paths

    def __len__(self) -> int:
        return len(self.sample_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = torch.load(self.sample_paths[idx], map_location="cpu", weights_only=False)
        potential = sample["input"]["potential"].float()  # [N,N]
        psi_t0 = sample["input"]["psi_t0"].float()  # [2,N,N]
        psi_t = sample["target"]["psi_t"].float()  # [2,N,N]

        x = torch.cat([potential.unsqueeze(0), psi_t0], dim=0)  # [3,N,N]
        y = psi_t  # [2,N,N]
        return x, y


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a compact 2D TDSE surrogate (UNet-like CNN)")
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="./tdse2d_runs")
    p.add_argument("--seed", type=int, default=1337)

    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-samples", type=int, default=0)

    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--physics-loss-weight", type=float, default=0.0)
    p.add_argument("--device", type=str, default="cpu")

    p.add_argument("--export-torchscript", action="store_true")
    p.add_argument("--torchscript-name", type=str, default="tdse2d_unet_torchscript.pt")
    return p.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_arg == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def collect_sample_paths(data_dir: str, max_samples: int) -> List[str]:
    paths = sorted(glob.glob(os.path.join(data_dir, "sample_*.pt")))
    if max_samples > 0:
        paths = paths[:max_samples]
    if not paths:
        raise RuntimeError(f"No sample_*.pt files found in {data_dir}")
    return paths


def load_dataset_meta(data_dir: str, fallback_grid: int) -> DatasetMeta:
    meta_path = os.path.join(data_dir, "dataset_meta.pt")
    if os.path.exists(meta_path):
        meta = torch.load(meta_path, map_location="cpu", weights_only=False)
        grid_size = int(meta.get("grid_size", fallback_grid))
        dx = float(meta.get("dx", 1.0 / max(1, grid_size)))
        return DatasetMeta(grid_size=grid_size, dx=dx)
    return DatasetMeta(grid_size=fallback_grid, dx=1.0 / max(1, fallback_grid))


def split_paths(paths: List[str], val_split: float) -> Tuple[List[str], List[str]]:
    n_total = len(paths)
    n_val = int(round(n_total * val_split))
    n_val = max(1, min(n_total - 1, n_val)) if n_total > 1 else 0
    val_paths = paths[-n_val:] if n_val > 0 else []
    train_paths = paths[:-n_val] if n_val > 0 else paths
    return train_paths, val_paths


class ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Tdse2dUNet(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 2, base: int = 32):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base * 2, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)
        self.head = nn.Conv2d(base, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))

        d2 = self.up2(b)
        if d2.shape[-2:] != e2.shape[-2:]:
            d2 = torch.nn.functional.interpolate(d2, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        if d1.shape[-2:] != e1.shape[-2:]:
            d1 = torch.nn.functional.interpolate(d1, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.head(d1)


def probability_norm(psi_ri: torch.Tensor, dx: float) -> torch.Tensor:
    prob_density = psi_ri[:, 0] ** 2 + psi_ri[:, 1] ** 2
    return prob_density.sum(dim=(1, 2)) * (dx ** 2)


def compute_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    input_state: torch.Tensor,
    mse_loss: nn.Module,
    physics_weight: float,
    dx: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    data_loss = mse_loss(pred, target)
    if physics_weight <= 0:
        zero = data_loss.new_tensor(0.0)
        return data_loss, data_loss, zero

    psi0 = input_state[:, 1:3]
    norm_in = probability_norm(psi0, dx)
    norm_pred = probability_norm(pred, dx)
    physics_loss = ((norm_pred - norm_in) ** 2).mean()
    total_loss = data_loss + physics_weight * physics_loss
    return total_loss, data_loss, physics_loss


def percent(done: int, total: int) -> float:
    if total <= 0:
        return 100.0
    return 100.0 * float(done) / float(total)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    mse_loss: nn.Module,
    physics_weight: float,
    dx: float,
    device: torch.device,
) -> Tuple[float, float, float]:
    model.eval()
    total = 0.0
    data_total = 0.0
    phys_total = 0.0
    count = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss, data_loss, phys_loss = compute_loss(pred, y, x, mse_loss, physics_weight, dx)
            b = x.shape[0]
            total += float(loss.item()) * b
            data_total += float(data_loss.item()) * b
            phys_total += float(phys_loss.item()) * b
            count += b
    if count == 0:
        return 0.0, 0.0, 0.0
    return total / count, data_total / count, phys_total / count


def export_to_torchscript(model: nn.Module, out_path: str, grid_size: int, device: torch.device) -> None:
    model.eval()
    dummy = torch.randn(1, 3, grid_size, grid_size, device=device)
    print("TorchScript export 0%: tracing", flush=True)
    with torch.no_grad():
        traced = torch.jit.trace(model, (dummy,), strict=False)
    traced = torch.jit.freeze(traced.eval())
    traced.save(out_path)
    print("TorchScript export 100%: finished", flush=True)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    all_paths = collect_sample_paths(args.data_dir, args.max_samples)
    train_paths, val_paths = split_paths(all_paths, args.val_split)

    train_ds = Tdse2dPairDataset(train_paths)
    val_ds = Tdse2dPairDataset(val_paths)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    first_x, _ = train_ds[0]
    grid_size = int(first_x.shape[-1])
    meta = load_dataset_meta(args.data_dir, fallback_grid=grid_size)

    model = Tdse2dUNet(in_ch=3, out_ch=2, base=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    mse_loss = nn.MSELoss()

    best_val = float("inf")
    best_ckpt_path = os.path.join(args.output_dir, "best_tdse2d_unet.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_total = 0.0
        running_data = 0.0
        running_phys = 0.0
        n_seen = 0

        n_batches = len(train_loader)
        progress_every = max(1, n_batches // 20)
        print(f"Epoch {epoch}/{args.epochs} 0%", flush=True)

        for batch_idx, (x, y) in enumerate(train_loader, start=1):
            x = x.to(device)
            y = y.to(device)
            pred = model(x)

            loss, data_loss, phys_loss = compute_loss(
                pred, y, x, mse_loss, args.physics_loss_weight, meta.dx
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            b = x.shape[0]
            running_total += float(loss.item()) * b
            running_data += float(data_loss.item()) * b
            running_phys += float(phys_loss.item()) * b
            n_seen += b

            if batch_idx % progress_every == 0 or batch_idx == n_batches:
                p = percent(batch_idx, n_batches)
                print(f"Epoch {epoch}/{args.epochs} {p:.1f}% (batch {batch_idx}/{n_batches})", flush=True)

        scheduler.step()

        train_total = running_total / max(1, n_seen)
        train_data = running_data / max(1, n_seen)
        train_phys = running_phys / max(1, n_seen)

        val_total, val_data, val_phys = evaluate(
            model, val_loader, mse_loss, args.physics_loss_weight, meta.dx, device
        )

        print(
            f"epoch={epoch:03d} "
            f"train_total={train_total:.6e} train_data={train_data:.6e} train_phys={train_phys:.6e} "
            f"val_total={val_total:.6e} val_data={val_data:.6e} val_phys={val_phys:.6e}"
        )

        if val_total < best_val:
            best_val = val_total
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "args": vars(args),
                    "dataset_meta": {"grid_size": meta.grid_size, "dx": meta.dx},
                    "best_val_total": best_val,
                    "epoch": epoch,
                },
                best_ckpt_path,
            )

    print("Best checkpoint:", best_ckpt_path)
    print(f"Best val_total={best_val:.6e}")

    if args.export_torchscript:
        ts_path = os.path.join(args.output_dir, args.torchscript_name)
        export_to_torchscript(model, ts_path, grid_size=meta.grid_size, device=device)
        print("Exported TorchScript model to", ts_path)


if __name__ == "__main__":
    main()
