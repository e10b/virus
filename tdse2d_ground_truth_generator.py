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
    p = argparse.ArgumentParser(description="Generate 2D TDSE training pairs with split-operator FFT")
    p.add_argument("--output-dir", type=str, default="./tdse2d_dataset")
    p.add_argument("--samples", type=int, default=200)
    p.add_argument("--grid-size", type=int, default=128, choices=[64, 96, 128, 192, 256])
    p.add_argument("--domain-length", type=float, default=24.0)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--target-step", type=int, default=20)
    p.add_argument("--mass", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--normalize-every", type=int, default=5)
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    return p.parse_args()


def make_coordinates(grid: GridConfig):
    x1d = np.linspace(-0.5 * grid.length, 0.5 * grid.length, grid.n, endpoint=False)
    return np.meshgrid(x1d, x1d, indexing="ij")


def make_k_squared(grid: GridConfig):
    k1d = 2.0 * math.pi * np.fft.fftfreq(grid.n, d=grid.dx)
    kx2 = k1d[:, None] ** 2
    ky2 = k1d[None, :] ** 2
    return kx2 + ky2


def random_wave_packet(rng: np.random.Generator, grid: GridConfig, complex_dtype):
    x, y = make_coordinates(grid)
    cx, cy = rng.uniform(-0.25 * grid.length, 0.25 * grid.length, size=(2,))
    sigma = rng.uniform(0.06 * grid.length, 0.14 * grid.length)
    px, py = rng.uniform(-2.5, 2.5, size=(2,))

    env = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * sigma * sigma))
    phase = px * x + py * y
    psi0 = env * np.exp(1j * phase)
    return psi0.astype(complex_dtype)


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


def random_potential(rng: np.random.Generator, grid: GridConfig, real_dtype):
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


def normalize_wavefunction(psi: np.ndarray, dx: float) -> np.ndarray:
    norm = np.sqrt(np.sum(np.abs(psi) ** 2) * (dx ** 2))
    if norm > 0:
        psi = psi / norm
    return psi


def evolve_soft_2d(
    psi0: np.ndarray,
    potential: np.ndarray,
    dt: float,
    steps: int,
    mass: float,
    dx: float,
    k_squared: np.ndarray,
    normalize_every: int,
    absorbing_mask: np.ndarray,
) -> np.ndarray:
    psi = psi0.copy()
    half_v = np.exp(-0.5j * dt * potential)
    kinetic = np.exp(-0.5j * dt * k_squared / mass)

    for step in range(steps):
        psi *= half_v
        psi_k = np.fft.fft2(psi)
        psi_k *= kinetic
        psi = np.fft.ifft2(psi_k)
        psi *= half_v
        psi *= absorbing_mask

        if normalize_every > 0 and ((step + 1) % normalize_every == 0):
            psi = normalize_wavefunction(psi, dx)

    return normalize_wavefunction(psi, dx)


def tensorize_sample(potential: np.ndarray, psi0: np.ndarray, psi_t: np.ndarray, meta: dict) -> dict:
    psi0_ch = np.stack([psi0.real, psi0.imag], axis=0).astype(np.float32)
    psit_ch = np.stack([psi_t.real, psi_t.imag], axis=0).astype(np.float32)
    return {
        "input": {
            "potential": torch.from_numpy(potential.astype(np.float32)),
            "psi_t0": torch.from_numpy(psi0_ch),
        },
        "target": {
            "psi_t": torch.from_numpy(psit_ch),
        },
        "meta": meta,
    }


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    grid = GridConfig(n=args.grid_size, length=args.domain_length)
    rng = np.random.default_rng(args.seed)

    complex_dtype = np.complex64 if args.dtype == "float32" else np.complex128
    real_dtype = np.float32 if args.dtype == "float32" else np.float64

    k_squared = make_k_squared(grid).astype(real_dtype)
    absorbing_mask = make_absorbing_mask(grid, real_dtype)

    dataset_meta = {
        "solver": "SOFT-2D",
        "equation": "i dpsi/dt = (-1/(2m) Laplacian + V) psi",
        "dimension": 2,
        "grid_size": args.grid_size,
        "domain_length": args.domain_length,
        "dx": grid.dx,
        "dt": args.dt,
        "target_step": args.target_step,
        "target_time": args.target_step * args.dt,
        "mass": args.mass,
        "normalize_every": args.normalize_every,
        "samples": args.samples,
        "seed": args.seed,
    }
    torch.save(dataset_meta, os.path.join(args.output_dir, "dataset_meta.pt"))

    for i in range(args.samples):
        psi0 = random_wave_packet(rng, grid, complex_dtype)
        psi0 = normalize_wavefunction(psi0, grid.dx)
        potential, p_kind = random_potential(rng, grid, real_dtype)

        psi_t = evolve_soft_2d(
            psi0=psi0,
            potential=potential,
            dt=args.dt,
            steps=args.target_step,
            mass=args.mass,
            dx=grid.dx,
            k_squared=k_squared,
            normalize_every=args.normalize_every,
            absorbing_mask=absorbing_mask,
        )

        sample = tensorize_sample(
            potential=potential,
            psi0=psi0,
            psi_t=psi_t,
            meta={
                "index": i,
                "potential_kind": p_kind,
                "target_step": args.target_step,
                "target_time": args.target_step * args.dt,
            },
        )

        out_path = os.path.join(args.output_dir, f"sample_{i:06d}.pt")
        torch.save(sample, out_path)

        if (i + 1) % 10 == 0 or i == 0 or (i + 1) == args.samples:
            print(f"[{i + 1:>6}/{args.samples}] saved {out_path} (potential={p_kind})")

    print("Done. 2D dataset written to", args.output_dir)


if __name__ == "__main__":
    main()
