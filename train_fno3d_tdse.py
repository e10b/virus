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


class TdsePairDataset(Dataset):
    def __init__(self, sample_paths: List[str]):
        self.sample_paths = sample_paths

    def __len__(self) -> int:
        return len(self.sample_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = torch.load(self.sample_paths[idx], map_location="cpu", weights_only=False)
        potential = sample["input"]["potential"].float()  # [N,N,N]
        psi_t0 = sample["input"]["psi_t0"].float()  # [2,N,N,N]
        psi_t = sample["target"]["psi_t"].float()  # [2,N,N,N]

        x = torch.cat([potential.unsqueeze(0), psi_t0], dim=0)  # [3,N,N,N]
        y = psi_t  # [2,N,N,N]
        return x, y


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a 3D Fourier Neural Operator (FNO3d) for TDSE state rollout and export artifacts."
    )
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./tdse_fno_runs")
    parser.add_argument("--seed", type=int, default=1337)

    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)

    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--fno-modes", type=int, default=16)
    parser.add_argument("--n-layers", type=int, default=4)

    parser.add_argument("--physics-loss-weight", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--export-onnx", action="store_true")
    parser.add_argument("--onnx-name", type=str, default="tdse_fno3d.onnx")
    parser.add_argument("--export-torchscript", action="store_true")
    parser.add_argument("--torchscript-name", type=str, default="tdse_fno3d_torchscript.pt")
    return parser.parse_args()


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


def build_fno3d(in_channels: int, out_channels: int, modes: int, hidden_channels: int, n_layers: int) -> nn.Module:
    try:
        from neuralop.models import FNO
    except Exception as exc:
        raise RuntimeError(
            "Could not import neuraloperator FNO model. Install it in your environment, e.g. `pip install neuraloperator`."
        ) from exc

    return FNO(
        n_modes=(modes, modes, modes),
        hidden_channels=hidden_channels,
        in_channels=in_channels,
        out_channels=out_channels,
        n_layers=n_layers,
    )


def probability_norm(psi_ri: torch.Tensor, dx: float) -> torch.Tensor:
    # psi_ri shape: [B,2,N,N,N]
    prob_density = psi_ri[:, 0] ** 2 + psi_ri[:, 1] ** 2
    return prob_density.sum(dim=(1, 2, 3)) * (dx ** 3)


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

    psi0 = input_state[:, 1:3]  # [B,2,N,N,N] (real, imag at t=0)
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


class _OnnxExportWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def state_dict(self, *args, **kwargs):
        raw = super().state_dict(*args, **kwargs)
        return {k: v for k, v in raw.items() if hasattr(v, "detach")}


def export_to_onnx(model: nn.Module, out_path: str, grid_size: int, device: torch.device) -> None:
    model.eval()
    export_model = _OnnxExportWrapper(model).to(device).eval()
    dummy = torch.randn(1, 3, grid_size, grid_size, grid_size, device=device)

    print("ONNX export 0%: preparing export", flush=True)

    def _short_exc(exc: Exception) -> str:
        msg = str(exc).replace("\n", " ").strip()
        if len(msg) > 400:
            msg = msg[:400] + " ...<truncated>"
        return f"{type(exc).__name__}: {msg}"

    # Prefer the newer exporter first (more robust on newer PyTorch/Python combos),
    # then fall back to legacy TorchScript exporter for compatibility.
    try:
        print("ONNX export 25%: trying dynamo exporter", flush=True)
        torch.onnx.export(
            export_model,
            (dummy,),
            out_path,
            input_names=["input_state"],
            output_names=["predicted_psi"],
            opset_version=17,
            do_constant_folding=True,
            dynamo=True,
        )
    except Exception as exc_dynamo:
        print(f"ONNX export fallback: dynamo failed ({_short_exc(exc_dynamo)})", flush=True)
        try:
            print("ONNX export 60%: trying legacy exporter", flush=True)

            traced = torch.jit.trace(export_model, (dummy,), strict=False)
            traced.eval()

            torch.onnx.export(
                traced,
                (dummy,),
                out_path,
                input_names=["input_state"],
                output_names=["predicted_psi"],
                opset_version=17,
                do_constant_folding=True,
                dynamo=False,
            )
        except Exception as exc_legacy:
            raise RuntimeError(
                "ONNX export failed. "
                f"Dynamo error: {_short_exc(exc_dynamo)} | "
                f"Legacy error: {_short_exc(exc_legacy)}"
            ) from None

    print("ONNX export 100%: finished", flush=True)


def export_to_torchscript(model: nn.Module, out_path: str, grid_size: int, device: torch.device) -> None:
    model.eval()
    export_model = _OnnxExportWrapper(model).to(device).eval()
    dummy = torch.randn(1, 3, grid_size, grid_size, grid_size, device=device)
    print("TorchScript export 0%: tracing", flush=True)
    with torch.no_grad():
        traced = torch.jit.trace(export_model, (dummy,), strict=False)
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

    train_ds = TdsePairDataset(train_paths)
    val_ds = TdsePairDataset(val_paths)

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

    model = build_fno3d(
        in_channels=3,
        out_channels=2,
        modes=args.fno_modes,
        hidden_channels=args.hidden_channels,
        n_layers=args.n_layers,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    mse_loss = nn.MSELoss()

    best_val = float("inf")
    best_ckpt_path = os.path.join(args.output_dir, "best_fno3d.pt")

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
                pred,
                y,
                x,
                mse_loss,
                args.physics_loss_weight,
                meta.dx,
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
                print(
                    f"Epoch {epoch}/{args.epochs} {p:.1f}% "
                    f"(batch {batch_idx}/{n_batches})",
                    flush=True,
                )

        scheduler.step()

        train_total = running_total / max(1, n_seen)
        train_data = running_data / max(1, n_seen)
        train_phys = running_phys / max(1, n_seen)

        val_total, val_data, val_phys = evaluate(
            model,
            val_loader,
            mse_loss,
            args.physics_loss_weight,
            meta.dx,
            device,
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

    if args.export_onnx:
        onnx_path = os.path.join(args.output_dir, args.onnx_name)
        export_to_onnx(model, onnx_path, grid_size=meta.grid_size, device=device)
        print("Exported ONNX model to", onnx_path)

    if args.export_torchscript:
        ts_path = os.path.join(args.output_dir, args.torchscript_name)
        export_to_torchscript(model, ts_path, grid_size=meta.grid_size, device=device)
        print("Exported TorchScript model to", ts_path)


if __name__ == "__main__":
    main()
