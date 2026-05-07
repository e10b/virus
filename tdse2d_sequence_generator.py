#!/usr/bin/env python3
import argparse
import math
import os
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class GridConfig:
    n: int
    length: float

    @property
    def dx(self) -> float:
        return self.length / self.n


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate 2D TDSE sequence data for autoregressive rollout training")
    p.add_argument("--output-dir", type=str, default="./tdse2d_sequence_dataset")
    p.add_argument("--samples", type=int, default=300)
    p.add_argument("--grid-size", type=int, default=96, choices=[64, 96, 128, 192, 256])
    p.add_argument("--domain-length", type=float, default=24.0)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--sequence-steps", type=int, default=32, help="Number of one-step transitions; sequence stores T+1 states")
    p.add_argument("--substeps-per-step", type=int, default=1, help="Internal SOFT substeps per saved step")
    p.add_argument("--mass", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--normalize-every", type=int, default=1)
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    p.add_argument("--backend", type=str, default="numpy", choices=["numpy", "torch"])
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument(
        "--potential-mode",
        type=str,
        default="mixed",
        choices=["mixed", "double_slit", "harmonic", "random_noise"],
        help="Potential family to generate. 'mixed' uses default blend.",
    )
    return p.parse_args()


def make_coordinates(grid: GridConfig):
    x1d = np.linspace(-0.5 * grid.length, 0.5 * grid.length, grid.n, endpoint=False)
    return np.meshgrid(x1d, x1d, indexing="ij")


def make_k_squared(grid: GridConfig):
    k1d = 2.0 * math.pi * np.fft.fftfreq(grid.n, d=grid.dx)
    return k1d[:, None] ** 2 + k1d[None, :] ** 2


def resolve_torch_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device_arg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_arg == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_k_squared_torch(grid: GridConfig, device: torch.device, real_dtype: torch.dtype) -> torch.Tensor:
    k1d = 2.0 * math.pi * torch.fft.fftfreq(grid.n, d=grid.dx, device=device, dtype=real_dtype)
    return k1d[:, None] ** 2 + k1d[None, :] ** 2


def normalize_wavefunction(psi: np.ndarray, dx: float) -> np.ndarray:
    norm = np.sqrt(np.sum(np.abs(psi) ** 2) * (dx ** 2))
    if norm > 0:
        psi = psi / norm
    return psi


def normalize_wavefunction_torch(psi: torch.Tensor, dx: float, eps: float = 1e-12) -> torch.Tensor:
    norm = torch.sqrt((psi.abs() ** 2).sum() * (dx ** 2)).clamp_min(eps)
    return psi / norm


def random_wave_packet(rng: np.random.Generator, grid: GridConfig, complex_dtype):
    x, y = make_coordinates(grid)
    cx, cy = rng.uniform(-0.25 * grid.length, 0.25 * grid.length, size=(2,))
    sigma = rng.uniform(0.06 * grid.length, 0.14 * grid.length)
    px, py = rng.uniform(-2.5, 2.5, size=(2,))

    env = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * sigma * sigma))
    phase = px * x + py * y
    psi = env * np.exp(1j * phase)
    return psi.astype(complex_dtype)


def potential_double_slit(rng: np.random.Generator, grid: GridConfig, real_dtype):
    x, y = make_coordinates(grid)
    barrier_h = rng.uniform(8.0, 18.0)
    barrier_w = rng.uniform(0.02 * grid.length, 0.05 * grid.length)
    slit_w = rng.uniform(0.04 * grid.length, 0.10 * grid.length)
    slit_sep = rng.uniform(0.12 * grid.length, 0.24 * grid.length)

    wall = np.abs(x) < barrier_w
    slit_a = np.abs(y - 0.5 * slit_sep) < slit_w
    slit_b = np.abs(y + 0.5 * slit_sep) < slit_w

    v = np.zeros((grid.n, grid.n), dtype=real_dtype)
    v = np.where(wall & (~(slit_a | slit_b)), barrier_h, v)
    return v.astype(real_dtype)


def potential_harmonic(rng: np.random.Generator, grid: GridConfig, real_dtype):
    x, y = make_coordinates(grid)
    wx = rng.uniform(0.08, 0.30)
    wy = rng.uniform(0.08, 0.30)
    cx, cy = rng.uniform(-0.15 * grid.length, 0.15 * grid.length, size=(2,))
    xr = x - cx
    yr = y - cy
    v = 0.5 * ((wx * xr) ** 2 + (wy * yr) ** 2)
    return v.astype(real_dtype)


def potential_random_noise(rng: np.random.Generator, grid: GridConfig, real_dtype):
    noise = rng.standard_normal((grid.n, grid.n)).astype(np.float32)
    k1d = 2.0 * math.pi * np.fft.fftfreq(grid.n, d=grid.dx)
    k2 = k1d[:, None] ** 2 + k1d[None, :] ** 2
    corr_len = rng.uniform(0.05 * grid.length, 0.16 * grid.length)
    filt = np.exp(-0.5 * corr_len * corr_len * k2)

    smooth = np.fft.ifft2(np.fft.fft2(noise) * filt).real
    smooth = smooth - smooth.mean()
    std = smooth.std()
    if std > 0:
        smooth = smooth / std
    amp = rng.uniform(0.5, 4.0)
    return (amp * smooth).astype(real_dtype)


def random_potential(rng: np.random.Generator, grid: GridConfig, real_dtype, potential_mode: str = "mixed"):
    if potential_mode == "double_slit":
        mode = "double_slit"
    elif potential_mode == "harmonic":
        mode = "harmonic"
    elif potential_mode == "random_noise":
        mode = "random_noise"
    else:
        mode = rng.choice(["double_slit", "harmonic", "random_noise"], p=[0.34, 0.33, 0.33])
    if mode == "double_slit":
        return potential_double_slit(rng, grid, real_dtype), mode
    if mode == "harmonic":
        return potential_harmonic(rng, grid, real_dtype), mode
    return potential_random_noise(rng, grid, real_dtype), mode


def make_absorbing_mask(grid: GridConfig, real_dtype):
    x, y = make_coordinates(grid)
    edge = 0.5 * grid.length
    absorb_start = 0.80 * edge
    absorb_strength = 1.5
    r = np.maximum(np.abs(x), np.abs(y))
    q = np.clip((r - absorb_start) / (edge - absorb_start + 1e-8), 0.0, 1.0)
    return np.exp(-absorb_strength * q * q).astype(real_dtype)


def soft_step(
    psi: np.ndarray,
    potential: np.ndarray,
    dt: float,
    mass: float,
    k_squared: np.ndarray,
    absorbing_mask: np.ndarray,
):
    half_v = np.exp(-0.5j * dt * potential)
    kinetic = np.exp(-0.5j * dt * k_squared / mass)

    psi = psi * half_v
    psi_k = np.fft.fft2(psi)
    psi_k = psi_k * kinetic
    psi = np.fft.ifft2(psi_k)
    psi = psi * half_v
    psi = psi * absorbing_mask
    return psi


def soft_step_torch(
    psi: torch.Tensor,
    potential: torch.Tensor,
    dt: float,
    mass: float,
    k_squared: torch.Tensor,
    absorbing_mask: torch.Tensor,
) -> torch.Tensor:
    half_v = torch.exp((-0.5j * dt) * potential)
    kinetic = torch.exp((-0.5j * dt / mass) * k_squared)

    psi = psi * half_v
    psi_k = torch.fft.fft2(psi)
    psi_k = psi_k * kinetic
    psi = torch.fft.ifft2(psi_k)
    psi = psi * half_v
    psi = psi * absorbing_mask
    return psi


def sequence_tensor(psi_seq_complex: np.ndarray) -> torch.Tensor:
    # psi_seq_complex: [T+1, N, N] complex
    psi_ri = np.stack([psi_seq_complex.real, psi_seq_complex.imag], axis=1).astype(np.float32)
    return torch.from_numpy(psi_ri)  # [T+1, 2, N, N]


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    grid = GridConfig(n=args.grid_size, length=args.domain_length)
    rng = np.random.default_rng(args.seed)

    complex_dtype = np.complex64 if args.dtype == "float32" else np.complex128
    real_dtype = np.float32 if args.dtype == "float32" else np.float64

    torch_real_dtype = torch.float32 if args.dtype == "float32" else torch.float64
    torch_complex_dtype = torch.complex64 if args.dtype == "float32" else torch.complex128

    torch_device = resolve_torch_device(args.device)
    if args.backend == "torch":
        print(f"Using torch backend on device: {torch_device}")
    else:
        print("Using numpy backend on CPU")

    k_squared = make_k_squared(grid).astype(real_dtype)
    absorbing_mask = make_absorbing_mask(grid, real_dtype)
    if args.backend == "torch":
        k_squared_t = make_k_squared_torch(grid, torch_device, torch_real_dtype)
        absorbing_mask_t = torch.as_tensor(absorbing_mask, dtype=torch_complex_dtype, device=torch_device)

    dataset_meta = {
        "solver": "SOFT-2D",
        "equation": "i dpsi/dt = (-1/(2m) Laplacian + V) psi",
        "dimension": 2,
        "grid_size": args.grid_size,
        "domain_length": args.domain_length,
        "dx": grid.dx,
        "dt": args.dt,
        "mass": args.mass,
        "normalize_every": args.normalize_every,
        "sequence_steps": args.sequence_steps,
        "substeps_per_step": args.substeps_per_step,
        "sequence_length": args.sequence_steps + 1,
        "samples": args.samples,
        "seed": args.seed,
        "backend": args.backend,
        "device": str(torch_device) if args.backend == "torch" else "cpu",
        "potential_mode": args.potential_mode,
    }
    torch.save(dataset_meta, os.path.join(args.output_dir, "dataset_meta.pt"))

    for i in range(args.samples):
        psi = random_wave_packet(rng, grid, complex_dtype)
        psi = normalize_wavefunction(psi, grid.dx)
        potential, potential_kind = random_potential(rng, grid, real_dtype, potential_mode=args.potential_mode)

        if args.backend == "torch":
            psi_t = torch.as_tensor(psi, dtype=torch_complex_dtype, device=torch_device)
            potential_t = torch.as_tensor(potential, dtype=torch_complex_dtype, device=torch_device)

            seq_t = torch.empty(
                (args.sequence_steps + 1, grid.n, grid.n),
                dtype=torch_complex_dtype,
                device=torch_device,
            )
            seq_t[0] = psi_t

            dt_sub = args.dt / max(1, args.substeps_per_step)
            for t in range(args.sequence_steps):
                for _ in range(max(1, args.substeps_per_step)):
                    psi_t = soft_step_torch(
                        psi=psi_t,
                        potential=potential_t,
                        dt=dt_sub,
                        mass=args.mass,
                        k_squared=k_squared_t,
                        absorbing_mask=absorbing_mask_t,
                    )
                if args.normalize_every > 0 and ((t + 1) % args.normalize_every == 0):
                    psi_t = normalize_wavefunction_torch(psi_t, grid.dx)
                seq_t[t + 1] = psi_t

            seq_np = seq_t.detach().cpu().numpy()
            psi_seq = sequence_tensor(seq_np)
        else:
            seq = np.zeros((args.sequence_steps + 1, grid.n, grid.n), dtype=complex_dtype)
            seq[0] = psi

            dt_sub = args.dt / max(1, args.substeps_per_step)
            for t in range(args.sequence_steps):
                for _ in range(max(1, args.substeps_per_step)):
                    psi = soft_step(
                        psi=psi,
                        potential=potential,
                        dt=dt_sub,
                        mass=args.mass,
                        k_squared=k_squared,
                        absorbing_mask=absorbing_mask,
                    )
                if args.normalize_every > 0 and ((t + 1) % args.normalize_every == 0):
                    psi = normalize_wavefunction(psi, grid.dx)
                seq[t + 1] = psi

            psi_seq = sequence_tensor(seq)
        sample = {
            "input": {
                "potential": torch.from_numpy(potential.astype(np.float32)),
                "psi_t0": psi_seq[0],
            },
            "target": {
                "psi_seq": psi_seq,
            },
            "meta": {
                "index": i,
                "potential_kind": potential_kind,
                "sequence_steps": args.sequence_steps,
                "dt": args.dt,
            },
        }

        out_path = os.path.join(args.output_dir, f"sample_{i:06d}.pt")
        torch.save(sample, out_path)

        if (i + 1) % 10 == 0 or i == 0 or (i + 1) == args.samples:
            print(f"[{i + 1:>6}/{args.samples}] saved {out_path} (potential={potential_kind})")

    print("Done. Sequence dataset written to", args.output_dir)


if __name__ == "__main__":
    main()
