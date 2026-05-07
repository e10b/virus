#!/usr/bin/env python3
import argparse
import glob
import math
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
    sequence_steps: int


class Tdse2dSequenceDataset(Dataset):
    def __init__(self, sample_paths: List[str]):
        self.sample_paths = sample_paths

    def __len__(self) -> int:
        return len(self.sample_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = torch.load(self.sample_paths[idx], map_location="cpu", weights_only=False)
        potential = sample["input"]["potential"].float()  # [N,N]
        psi_seq = sample["target"]["psi_seq"].float()  # [T+1,2,N,N]
        return potential, psi_seq


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train 2D TDSE delta model with autoregressive rollout loss")
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="./tdse2d_delta_runs")
    p.add_argument("--seed", type=int, default=1337)

    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-samples", type=int, default=0)

    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument(
        "--fourier-levels",
        type=int,
        default=0,
        help="If >0, append Fourier coordinate features [x,y,sin/cos(2^l*w*x),sin/cos(2^l*w*y)]",
    )
    p.add_argument(
        "--fourier-base-freq",
        type=float,
        default=math.pi,
        help="Base angular frequency w for Fourier coordinate features",
    )
    p.add_argument(
        "--fourier-level-amp-decay",
        type=float,
        default=0.7,
        help="Per-level amplitude decay for Fourier features; <1.0 reduces high-frequency aggressiveness",
    )
    p.add_argument("--rollout-steps", type=int, default=8, help="Training rollout horizon (<= dataset sequence_steps)")
    p.add_argument(
        "--rollout-curriculum-start",
        type=int,
        default=2,
        help="Starting rollout horizon for curriculum (ramps to --rollout-steps)",
    )
    p.add_argument(
        "--rollout-curriculum-epochs",
        type=int,
        default=0,
        help="Number of epochs to ramp rollout horizon; 0 disables curriculum",
    )
    p.add_argument(
        "--horizon-loss-power",
        type=float,
        default=2.0,
        help="Per-step loss weight power; >1 emphasizes late rollout steps",
    )
    p.add_argument("--physics-loss-weight", type=float, default=0.1, help="Weight for norm conservation penalty")
    p.add_argument("--renorm-each-step", action="store_true", help="Re-normalize psi after each predicted step")
    p.add_argument(
        "--random-start-prob",
        type=float,
        default=1.0,
        help="Probability of starting each training rollout from a random sequence time index",
    )
    p.add_argument(
        "--state-noise-std",
        type=float,
        default=0.003,
        help="Gaussian noise std injected into current state during training rollout (0 disables)",
    )
    p.add_argument(
        "--state-noise-decay-epochs",
        type=int,
        default=20,
        help="If >0, linearly decay state noise to 0 over this many epochs",
    )
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--resume-checkpoint", type=str, default="", help="Resume training from checkpoint path")

    p.add_argument("--export-torchscript", action="store_true")
    p.add_argument("--torchscript-name", type=str, default="tdse2d_delta_torchscript.pt")
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


def load_dataset_meta(data_dir: str, fallback_grid: int, fallback_steps: int) -> DatasetMeta:
    meta_path = os.path.join(data_dir, "dataset_meta.pt")
    if os.path.exists(meta_path):
        meta = torch.load(meta_path, map_location="cpu", weights_only=False)
        grid_size = int(meta.get("grid_size", fallback_grid))
        dx = float(meta.get("dx", 1.0 / max(1, grid_size)))
        seq_steps = int(meta.get("sequence_steps", fallback_steps))
        return DatasetMeta(grid_size=grid_size, dx=dx, sequence_steps=seq_steps)
    return DatasetMeta(grid_size=fallback_grid, dx=1.0 / max(1, fallback_grid), sequence_steps=fallback_steps)


def fourier_feature_channels(fourier_levels: int) -> int:
    levels = max(0, int(fourier_levels))
    if levels <= 0:
        return 0
    return 2 + 4 * levels


def build_fourier_coord_features(
    batch_size: int,
    n: int,
    levels: int,
    base_freq: float,
    device: torch.device,
    dtype: torch.dtype,
    level_amp_decay: float = 1.0,
) -> torch.Tensor | None:
    levels = max(0, int(levels))
    if levels <= 0:
        return None

    if n <= 1:
        coord = torch.zeros((n,), device=device, dtype=dtype)
    else:
        coord = torch.linspace(-1.0, 1.0, steps=n, device=device, dtype=dtype)
    x, y = torch.meshgrid(coord, coord, indexing="ij")

    x = x.unsqueeze(0).unsqueeze(0)
    y = y.unsqueeze(0).unsqueeze(0)
    feats = [x, y]

    w0 = float(base_freq)
    amp_decay = float(level_amp_decay)
    for l in range(levels):
        w = (2.0 ** float(l)) * w0
        amp = amp_decay ** float(l)
        feats.append(amp * torch.sin(w * x))
        feats.append(amp * torch.cos(w * x))
        feats.append(amp * torch.sin(w * y))
        feats.append(amp * torch.cos(w * y))

    f = torch.cat(feats, dim=1)
    return f.expand(batch_size, -1, -1, -1)


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


class DeltaUNet2D(nn.Module):
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
    # psi_ri: [B,2,N,N]
    rho = psi_ri[:, 0] ** 2 + psi_ri[:, 1] ** 2
    return rho.sum(dim=(1, 2)) * (dx ** 2)


def renormalize_psi(psi_ri: torch.Tensor, dx: float, eps: float = 1e-12) -> torch.Tensor:
    norm = probability_norm(psi_ri, dx).clamp_min(eps).sqrt()  # [B]
    return psi_ri / norm[:, None, None, None]


def rollout_loss(
    model: nn.Module,
    potential: torch.Tensor,
    psi_seq: torch.Tensor,
    rollout_steps: int,
    mse_loss: nn.Module,
    physics_weight: float,
    dx: float,
    renorm_each_step: bool,
    horizon_loss_power: float,
    fourier_levels: int,
    fourier_base_freq: float,
    fourier_level_amp_decay: float,
    random_start: bool,
    state_noise_std: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # potential: [B,N,N], psi_seq: [B,T+1,2,N,N]
    b, t_plus_1, _, n, _ = psi_seq.shape
    seq_steps = t_plus_1 - 1
    max_steps = min(rollout_steps, seq_steps)
    start_idx_max = max(0, seq_steps - max_steps)
    if random_start and start_idx_max > 0:
        start_idx = int(torch.randint(0, start_idx_max + 1, (1,), device=psi_seq.device).item())
    else:
        start_idx = 0

    psi_cur = psi_seq[:, start_idx]  # [B,2,N,N]
    total_data = psi_cur.new_tensor(0.0)
    total_phys = psi_cur.new_tensor(0.0)
    weight_sum = psi_cur.new_tensor(0.0)
    coord_feats = build_fourier_coord_features(
        batch_size=b,
        n=n,
        levels=fourier_levels,
        base_freq=fourier_base_freq,
        level_amp_decay=fourier_level_amp_decay,
        device=psi_seq.device,
        dtype=psi_seq.dtype,
    )

    for t in range(max_steps):
        w_scalar = ((float(t + 1) / float(max_steps)) ** max(0.0, float(horizon_loss_power))) if max_steps > 0 else 1.0
        w = psi_cur.new_tensor(w_scalar)
        if state_noise_std > 0:
            psi_in = psi_cur + (float(state_noise_std) * torch.randn_like(psi_cur))
            psi_in = renormalize_psi(psi_in, dx)
        else:
            psi_in = psi_cur

        inp = torch.cat([potential.unsqueeze(1), psi_in], dim=1)  # [B,3,N,N]
        if coord_feats is not None:
            inp = torch.cat([inp, coord_feats], dim=1)
        delta = model(inp)
        psi_next = psi_in + delta
        if renorm_each_step:
            psi_next = renormalize_psi(psi_next, dx)

        gt_next = psi_seq[:, start_idx + t + 1]
        total_data = total_data + w * mse_loss(psi_next, gt_next)

        if physics_weight > 0:
            norm_next = probability_norm(psi_next, dx)
            total_phys = total_phys + w * ((norm_next - 1.0) ** 2).mean()

        weight_sum = weight_sum + w

        psi_cur = psi_next

    denom = weight_sum.clamp_min(1e-12)
    data_loss = total_data / denom
    phys_loss = total_phys / denom if physics_weight > 0 else total_data.new_tensor(0.0)
    total_loss = data_loss + physics_weight * phys_loss
    return total_loss, data_loss, phys_loss


def percent(done: int, total: int) -> float:
    if total <= 0:
        return 100.0
    return 100.0 * float(done) / float(total)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    rollout_steps: int,
    mse_loss: nn.Module,
    physics_weight: float,
    dx: float,
    renorm_each_step: bool,
    horizon_loss_power: float,
    fourier_levels: int,
    fourier_base_freq: float,
    fourier_level_amp_decay: float,
    device: torch.device,
) -> Tuple[float, float, float]:
    model.eval()
    total = 0.0
    data_total = 0.0
    phys_total = 0.0
    count = 0

    with torch.no_grad():
        for potential, psi_seq in loader:
            potential = potential.to(device)
            psi_seq = psi_seq.to(device)
            loss, data_loss, phys_loss = rollout_loss(
                model=model,
                potential=potential,
                psi_seq=psi_seq,
                rollout_steps=rollout_steps,
                mse_loss=mse_loss,
                physics_weight=physics_weight,
                dx=dx,
                renorm_each_step=renorm_each_step,
                horizon_loss_power=horizon_loss_power,
                fourier_levels=fourier_levels,
                fourier_base_freq=fourier_base_freq,
                fourier_level_amp_decay=fourier_level_amp_decay,
                random_start=False,
                state_noise_std=0.0,
            )
            b = potential.shape[0]
            total += float(loss.item()) * b
            data_total += float(data_loss.item()) * b
            phys_total += float(phys_loss.item()) * b
            count += b

    if count == 0:
        return 0.0, 0.0, 0.0
    return total / count, data_total / count, phys_total / count


class DeltaStepWrapper(nn.Module):
    def __init__(self, model: nn.Module, dx: float, renorm_each_step: bool):
        super().__init__()
        self.model = model
        self.dx = float(dx)
        self.renorm_each_step = bool(renorm_each_step)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,3,N,N] = [potential, psi_re, psi_im]
        psi_cur = x[:, 1:3]
        delta = self.model(x)
        psi_next = psi_cur + delta
        if self.renorm_each_step:
            psi_next = renormalize_psi(psi_next, self.dx)
        return psi_next


def export_to_torchscript(model: nn.Module, out_path: str, grid_size: int, device: torch.device, dx: float, renorm_each_step: bool) -> None:
    model.eval()
    wrapper = DeltaStepWrapper(model, dx=dx, renorm_each_step=renorm_each_step).to(device).eval()
    in_ch = int(model.enc1.net[0].in_channels)
    dummy = torch.randn(1, in_ch, grid_size, grid_size, device=device)
    print("TorchScript export 0%: tracing", flush=True)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (dummy,), strict=False)
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

    train_ds = Tdse2dSequenceDataset(train_paths)
    val_ds = Tdse2dSequenceDataset(val_paths)

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

    first_potential, first_seq = train_ds[0]
    grid_size = int(first_potential.shape[-1])
    dataset_steps = int(first_seq.shape[0] - 1)

    meta = load_dataset_meta(args.data_dir, fallback_grid=grid_size, fallback_steps=dataset_steps)
    rollout_steps = min(int(args.rollout_steps), int(meta.sequence_steps))
    curriculum_start = max(1, min(int(args.rollout_curriculum_start), rollout_steps))
    curriculum_epochs = max(0, int(args.rollout_curriculum_epochs))
    if args.fourier_levels < 0:
        raise RuntimeError("--fourier-levels must be >= 0")
    if args.fourier_level_amp_decay <= 0:
        raise RuntimeError("--fourier-level-amp-decay must be > 0")
    if args.random_start_prob < 0 or args.random_start_prob > 1:
        raise RuntimeError("--random-start-prob must be in [0,1]")
    if args.state_noise_std < 0:
        raise RuntimeError("--state-noise-std must be >= 0")
    if args.state_noise_decay_epochs < 0:
        raise RuntimeError("--state-noise-decay-epochs must be >= 0")
    in_ch = 3 + fourier_feature_channels(args.fourier_levels)

    model = DeltaUNet2D(in_ch=in_ch, out_ch=2, base=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    mse_loss = nn.MSELoss()

    best_val = float("inf")
    best_ckpt_path = os.path.join(args.output_dir, "best_tdse2d_delta.pt")
    start_epoch = 1

    if args.resume_checkpoint:
        resume = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(resume["model_state_dict"], strict=True)
        if "optimizer_state_dict" in resume:
            optimizer.load_state_dict(resume["optimizer_state_dict"])
        if "scheduler_state_dict" in resume:
            scheduler.load_state_dict(resume["scheduler_state_dict"])
        best_val = float(resume.get("best_val_total", best_val))
        last_epoch = int(resume.get("epoch", 0))
        start_epoch = last_epoch + 1
        print(
            f"Resumed from {args.resume_checkpoint} (last_epoch={last_epoch}, best_val={best_val:.6e})",
            flush=True,
        )
        if start_epoch > args.epochs:
            print(
                f"Nothing to do: start_epoch={start_epoch} exceeds --epochs={args.epochs}. Increase --epochs to continue.",
                flush=True,
            )
            return

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_total = 0.0
        running_data = 0.0
        running_phys = 0.0
        n_seen = 0

        n_batches = len(train_loader)
        progress_every = max(1, n_batches // 20)
        if curriculum_epochs > 0:
            frac = min(1.0, float(epoch - 1) / float(max(1, curriculum_epochs - 1)))
            current_rollout_steps = int(round(curriculum_start + frac * (rollout_steps - curriculum_start)))
        else:
            current_rollout_steps = rollout_steps
        current_rollout_steps = max(1, min(current_rollout_steps, rollout_steps))

        if args.state_noise_decay_epochs > 0:
            noise_frac = max(0.0, 1.0 - float(epoch - 1) / float(args.state_noise_decay_epochs))
            current_state_noise_std = float(args.state_noise_std) * noise_frac
        else:
            current_state_noise_std = float(args.state_noise_std)

        print(
            f"Epoch {epoch}/{args.epochs} 0% (rollout_steps={current_rollout_steps}, horizon_power={args.horizon_loss_power:.2f}, noise_std={current_state_noise_std:.2e})",
            flush=True,
        )

        for batch_idx, (potential, psi_seq) in enumerate(train_loader, start=1):
            potential = potential.to(device)
            psi_seq = psi_seq.to(device)

            loss, data_loss, phys_loss = rollout_loss(
                model=model,
                potential=potential,
                psi_seq=psi_seq,
                rollout_steps=current_rollout_steps,
                mse_loss=mse_loss,
                physics_weight=args.physics_loss_weight,
                dx=meta.dx,
                renorm_each_step=args.renorm_each_step,
                horizon_loss_power=args.horizon_loss_power,
                fourier_levels=args.fourier_levels,
                fourier_base_freq=args.fourier_base_freq,
                fourier_level_amp_decay=args.fourier_level_amp_decay,
                random_start=(random.random() < float(args.random_start_prob)),
                state_noise_std=current_state_noise_std,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            b = potential.shape[0]
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
            model=model,
            loader=val_loader,
            rollout_steps=rollout_steps,
            mse_loss=mse_loss,
            physics_weight=args.physics_loss_weight,
            dx=meta.dx,
            renorm_each_step=args.renorm_each_step,
            horizon_loss_power=args.horizon_loss_power,
            fourier_levels=args.fourier_levels,
            fourier_base_freq=args.fourier_base_freq,
            fourier_level_amp_decay=args.fourier_level_amp_decay,
            device=device,
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
                    "scheduler_state_dict": scheduler.state_dict(),
                    "args": vars(args),
                    "dataset_meta": {
                        "grid_size": meta.grid_size,
                        "dx": meta.dx,
                        "sequence_steps": meta.sequence_steps,
                    },
                    "best_val_total": best_val,
                    "epoch": epoch,
                },
                best_ckpt_path,
            )

    print("Best checkpoint:", best_ckpt_path)
    print(f"Best val_total={best_val:.6e}")

    if args.export_torchscript:
        ts_path = os.path.join(args.output_dir, args.torchscript_name)
        export_to_torchscript(
            model=model,
            out_path=ts_path,
            grid_size=meta.grid_size,
            device=device,
            dx=meta.dx,
            renorm_each_step=args.renorm_each_step,
        )
        print("Exported TorchScript delta-step model to", ts_path)


if __name__ == "__main__":
    main()
