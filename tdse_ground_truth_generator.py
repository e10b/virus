#!/usr/bin/env python3
import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict

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
    parser = argparse.ArgumentParser(
        description="Generate TDSE ground-truth training pairs with Split-Operator Fourier Transform (SOFT)."
    )
    parser.add_argument("--output-dir", type=str, default="./tdse_dataset")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--grid-size", type=int, default=128, choices=[64, 96, 128, 192, 256])
    parser.add_argument("--domain-length", type=float, default=24.0)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--target-step", type=int, default=100)
    parser.add_argument("--mass", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--backend", type=str, default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--normalize-every", type=int, default=10)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument("--benchmark-ho", action="store_true")
    parser.add_argument("--benchmark-ho-omega", type=float, default=0.25)
    parser.add_argument("--benchmark-ho-steps", type=int, default=200)
    parser.add_argument("--benchmark-ho-plot", type=str, default="harmonic_residual_error.png")
    return parser.parse_args()


def resolve_backend(backend: str):
    if backend == "cpu":
        return np, np.fft, None

    try:
        import cupy as cp
        import cupyx.scipy.fft as cp_fft
    except Exception as exc:
        raise RuntimeError(
            "GPU backend requested but CuPy/cupyx is unavailable. Install cupy first or use --backend cpu."
        ) from exc

    return cp, cp_fft, cp


def make_coordinates(xp, grid: GridConfig):
    x1d = xp.linspace(-0.5 * grid.length, 0.5 * grid.length, grid.n, endpoint=False)
    return xp.meshgrid(x1d, x1d, x1d, indexing="ij")


def make_k_squared(xp, grid: GridConfig):
    k1d = 2.0 * math.pi * xp.fft.fftfreq(grid.n, d=grid.dx)
    kx2 = k1d[:, None, None] ** 2
    ky2 = k1d[None, :, None] ** 2
    kz2 = k1d[None, None, :] ** 2
    return kx2 + ky2 + kz2


def random_wave_packet(xp, rng_np: np.random.Generator, grid: GridConfig, dtype):
    x, y, z = make_coordinates(xp, grid)

    center = rng_np.uniform(-0.25 * grid.length, 0.25 * grid.length, size=(3,))
    sigma = rng_np.uniform(0.06 * grid.length, 0.14 * grid.length)
    momentum = rng_np.uniform(-2.0, 2.0, size=(3,))

    xc = x - center[0]
    yc = y - center[1]
    zc = z - center[2]

    envelope = xp.exp(-(xc * xc + yc * yc + zc * zc) / (2.0 * sigma * sigma))
    phase = momentum[0] * x + momentum[1] * y + momentum[2] * z

    psi0 = envelope * xp.exp(1j * phase)
    psi0 = psi0.astype(dtype)
    return psi0


def potential_double_slit(xp, rng_np: np.random.Generator, grid: GridConfig, dtype):
    x, y, z = make_coordinates(xp, grid)
    barrier_height = rng_np.uniform(8.0, 18.0)
    barrier_width = rng_np.uniform(0.02 * grid.length, 0.06 * grid.length)
    slit_width = rng_np.uniform(0.04 * grid.length, 0.10 * grid.length)
    slit_sep = rng_np.uniform(0.12 * grid.length, 0.24 * grid.length)

    wall = xp.abs(x) < barrier_width
    slit_a = xp.abs(y - 0.5 * slit_sep) < slit_width
    slit_b = xp.abs(y + 0.5 * slit_sep) < slit_width
    slits = slit_a | slit_b

    v = xp.zeros((grid.n, grid.n, grid.n), dtype=dtype)
    v = xp.where(wall & (~slits), barrier_height, v)

    z_mod = rng_np.uniform(0.0, 0.15)
    if z_mod > 0:
        v = v * (1.0 + z_mod * xp.cos(2.0 * math.pi * z / grid.length))

    return v.astype(dtype)


def potential_harmonic(xp, rng_np: np.random.Generator, grid: GridConfig, dtype):
    x, y, z = make_coordinates(xp, grid)

    wx = rng_np.uniform(0.10, 0.35)
    wy = rng_np.uniform(0.10, 0.35)
    wz = rng_np.uniform(0.10, 0.35)

    cx, cy, cz = rng_np.uniform(-0.15 * grid.length, 0.15 * grid.length, size=(3,))
    xr = x - cx
    yr = y - cy
    zr = z - cz

    v = 0.5 * ((wx * xr) ** 2 + (wy * yr) ** 2 + (wz * zr) ** 2)
    return v.astype(dtype)


def potential_random_noise(xp, rng_np: np.random.Generator, grid: GridConfig, dtype):
    noise = rng_np.standard_normal((grid.n, grid.n, grid.n)).astype(np.float32)
    noise_xp = xp.asarray(noise)

    k1d = 2.0 * math.pi * xp.fft.fftfreq(grid.n, d=grid.dx)
    k2 = k1d[:, None, None] ** 2 + k1d[None, :, None] ** 2 + k1d[None, None, :] ** 2

    corr_len = rng_np.uniform(0.06 * grid.length, 0.16 * grid.length)
    filter_k = xp.exp(-0.5 * (corr_len * corr_len) * k2)

    noise_k = xp.fft.fftn(noise_xp)
    smooth = xp.fft.ifftn(noise_k * filter_k).real

    amp = rng_np.uniform(0.5, 4.0)
    smooth = smooth - xp.mean(smooth)
    std = xp.std(smooth)
    if float(std) > 0:
        smooth = smooth / std

    v = amp * smooth
    return v.astype(dtype)


def random_potential(xp, rng_np: np.random.Generator, grid: GridConfig, dtype):
    mode = rng_np.choice(["double_slit", "harmonic", "random_noise"], p=[0.34, 0.33, 0.33])
    if mode == "double_slit":
        return potential_double_slit(xp, rng_np, grid, dtype), mode
    if mode == "harmonic":
        return potential_harmonic(xp, rng_np, grid, dtype), mode
    return potential_random_noise(xp, rng_np, grid, dtype), mode


def make_absorbing_mask(xp, grid: GridConfig, dtype):
    x, y, z = make_coordinates(xp, grid)
    edge = 0.5 * grid.length
    absorb_start = 0.80 * edge
    absorb_strength = 1.5

    rx = xp.abs(x)
    ry = xp.abs(y)
    rz = xp.abs(z)
    r = xp.maximum(rx, xp.maximum(ry, rz))

    q = xp.clip((r - absorb_start) / (edge - absorb_start + 1e-8), 0.0, 1.0)
    return xp.exp(-absorb_strength * q * q).astype(dtype)


def normalize_wavefunction(xp, psi, dx: float):
    prob = xp.abs(psi) ** 2
    norm = xp.sqrt(xp.sum(prob) * (dx ** 3))
    nrm = float(norm)
    if nrm > 0:
        psi = psi / norm
    return psi


def evolve_soft(
    xp,
    fft_mod,
    psi0,
    potential,
    dt: float,
    steps: int,
    mass: float,
    dx: float,
    k_squared,
    normalize_every: int,
    absorbing_mask,
):
    psi = psi0.copy()

    half_v = xp.exp(-0.5j * dt * potential)
    kinetic = xp.exp(-0.5j * dt * k_squared / mass)

    for step in range(steps):
        psi *= half_v
        psi_k = fft_mod.fftn(psi)
        psi_k *= kinetic
        psi = fft_mod.ifftn(psi_k)
        psi *= half_v
        psi *= absorbing_mask

        if normalize_every > 0 and ((step + 1) % normalize_every == 0):
            psi = normalize_wavefunction(xp, psi, dx)

    return normalize_wavefunction(xp, psi, dx)


def tensorize_sample(potential, psi0, psi_t, meta: Dict, xp):
    if hasattr(xp, "asnumpy"):
        potential_np = xp.asnumpy(potential)
        psi0_np = xp.asnumpy(psi0)
        psi_t_np = xp.asnumpy(psi_t)
    else:
        potential_np = np.asarray(potential)
        psi0_np = np.asarray(psi0)
        psi_t_np = np.asarray(psi_t)

    psi0_ch = np.stack([psi0_np.real, psi0_np.imag], axis=0).astype(np.float32)
    psit_ch = np.stack([psi_t_np.real, psi_t_np.imag], axis=0).astype(np.float32)

    return {
        "input": {
            "potential": torch.from_numpy(potential_np.astype(np.float32)),
            "psi_t0": torch.from_numpy(psi0_ch),
        },
        "target": {
            "psi_t": torch.from_numpy(psit_ch),
        },
        "meta": meta,
    }


def harmonic_ground_state(xp, grid: GridConfig, mass: float, omega: float, dtype):
    x, y, z = make_coordinates(xp, grid)
    r2 = x * x + y * y + z * z
    pref = (mass * omega / math.pi) ** 0.75
    psi = pref * xp.exp(-0.5 * mass * omega * r2)
    return psi.astype(dtype)


def phase_aligned_relative_error(xp, psi_ref, psi_num, dx: float):
    overlap = xp.sum(xp.conj(psi_ref) * psi_num) * (dx ** 3)
    overlap_abs = float(xp.abs(overlap))
    if overlap_abs > 0:
        phase = xp.exp(-1j * xp.angle(overlap))
        psi_num = psi_num * phase

    diff = psi_num - psi_ref
    num = xp.sqrt(xp.sum(xp.abs(diff) ** 2) * (dx ** 3))
    den = xp.sqrt(xp.sum(xp.abs(psi_ref) ** 2) * (dx ** 3))
    rel_l2 = float(num / den) if float(den) > 0 else float("nan")
    max_abs = float(xp.max(xp.abs(diff)))
    return rel_l2, max_abs


def run_harmonic_benchmark(args, xp, fft_mod, complex_dtype, real_dtype):
    grid = GridConfig(n=args.grid_size, length=args.domain_length)
    k_squared = make_k_squared(xp, grid).astype(real_dtype)

    omega = args.benchmark_ho_omega
    x, y, z = make_coordinates(xp, grid)
    potential = 0.5 * args.mass * (omega ** 2) * (x * x + y * y + z * z)
    potential = potential.astype(real_dtype)

    psi0 = harmonic_ground_state(xp, grid, args.mass, omega, complex_dtype)
    psi0 = normalize_wavefunction(xp, psi0, grid.dx)
    psi = psi0.copy()

    half_v = xp.exp(-0.5j * args.dt * potential)
    kinetic = xp.exp(-0.5j * args.dt * k_squared / args.mass)

    energy = 1.5 * omega
    times = []
    rel_l2_errors = []
    max_abs_errors = []

    for step in range(args.benchmark_ho_steps):
        psi *= half_v
        psi_k = fft_mod.fftn(psi)
        psi_k *= kinetic
        psi = fft_mod.ifftn(psi_k)
        psi *= half_v

        if args.normalize_every > 0 and ((step + 1) % args.normalize_every == 0):
            psi = normalize_wavefunction(xp, psi, grid.dx)

        t = (step + 1) * args.dt
        psi_exact = psi0 * xp.exp(-1j * energy * t)
        rel_l2, max_abs = phase_aligned_relative_error(xp, psi_exact, psi, grid.dx)

        times.append(t)
        rel_l2_errors.append(rel_l2)
        max_abs_errors.append(max_abs)

    metrics = {
        "times": torch.tensor(times, dtype=torch.float64),
        "rel_l2_error": torch.tensor(rel_l2_errors, dtype=torch.float64),
        "max_abs_error": torch.tensor(max_abs_errors, dtype=torch.float64),
        "grid_size": args.grid_size,
        "domain_length": args.domain_length,
        "dt": args.dt,
        "steps": args.benchmark_ho_steps,
        "mass": args.mass,
        "omega": omega,
        "backend": args.backend,
    }

    metrics_path = os.path.join(args.output_dir, "harmonic_residual_metrics.pt")
    torch.save(metrics, metrics_path)

    plot_path = os.path.join(args.output_dir, args.benchmark_ho_plot)
    profile_plot_path = os.path.join(args.output_dir, "harmonic_profile_comparison.png")
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 5))
        plt.plot(times, rel_l2_errors, label="Relative L2 error")
        plt.plot(times, max_abs_errors, label="Max abs error")
        plt.xlabel("Time")
        plt.ylabel("Error")
        plt.title("SOFT vs analytic 3D harmonic oscillator")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print("Saved benchmark plot to", plot_path)

        if hasattr(xp, "asnumpy"):
            x_np = xp.asnumpy(x[:, grid.n // 2, grid.n // 2])
            psi_num_np = xp.asnumpy(psi[:, grid.n // 2, grid.n // 2])
            psi_exact_np = xp.asnumpy(psi_exact[:, grid.n // 2, grid.n // 2])
        else:
            x_np = np.asarray(x[:, grid.n // 2, grid.n // 2])
            psi_num_np = np.asarray(psi[:, grid.n // 2, grid.n // 2])
            psi_exact_np = np.asarray(psi_exact[:, grid.n // 2, grid.n // 2])

        rho_num = np.abs(psi_num_np) ** 2
        rho_exact = np.abs(psi_exact_np) ** 2

        plt.figure(figsize=(8, 5))
        plt.plot(x_np, rho_exact, label="Analytic harmonic |ψ|²", linewidth=2.0)
        plt.plot(
            x_np,
            rho_num,
            label="Our SOFT |ψ|²",
            linewidth=2.0,
            linestyle=":",
        )
        plt.xlabel("x (center-line slice at y=0, z=0)")
        plt.ylabel("Probability density |ψ|²")
        plt.title(f"Physical profile comparison at t={times[-1]:.3f}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(profile_plot_path, dpi=150)
        plt.close()
        print("Saved profile comparison plot to", profile_plot_path)
    except Exception as exc:
        print("Could not generate plot (matplotlib unavailable or failed):", exc)

    print("Saved benchmark metrics to", metrics_path)
    print(
        "Final errors:",
        f"rel_l2={rel_l2_errors[-1]:.6e}, max_abs={max_abs_errors[-1]:.6e}",
    )


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    xp, fft_mod, cp = resolve_backend(args.backend)
    dtype = np.complex64 if args.dtype == "float32" else np.complex128
    real_dtype = np.float32 if args.dtype == "float32" else np.float64

    if args.benchmark_ho:
        run_harmonic_benchmark(args, xp, fft_mod, dtype, real_dtype)
        if args.backend == "gpu" and cp is not None:
            cp.cuda.Stream.null.synchronize()
        return

    grid = GridConfig(n=args.grid_size, length=args.domain_length)
    rng = np.random.default_rng(args.seed)

    k_squared = make_k_squared(xp, grid).astype(real_dtype)
    absorbing_mask = make_absorbing_mask(xp, grid, real_dtype)

    dataset_meta = {
        "solver": "SOFT",
        "equation": "i dpsi/dt = (-1/(2m) Laplacian + V) psi",
        "grid_size": args.grid_size,
        "domain_length": args.domain_length,
        "dx": grid.dx,
        "dt": args.dt,
        "target_step": args.target_step,
        "target_time": args.target_step * args.dt,
        "mass": args.mass,
        "backend": args.backend,
        "normalize_every": args.normalize_every,
        "samples": args.samples,
        "seed": args.seed,
    }
    torch.save(dataset_meta, os.path.join(args.output_dir, "dataset_meta.pt"))

    for i in range(args.samples):
        psi0 = random_wave_packet(xp, rng, grid, dtype)
        psi0 = normalize_wavefunction(xp, psi0, grid.dx)

        potential, potential_kind = random_potential(xp, rng, grid, real_dtype)
        psi_t = evolve_soft(
            xp=xp,
            fft_mod=fft_mod,
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

        sample_meta = {
            "index": i,
            "potential_kind": potential_kind,
            "target_step": args.target_step,
            "target_time": args.target_step * args.dt,
        }
        sample = tensorize_sample(potential, psi0, psi_t, sample_meta, xp)

        out_path = os.path.join(args.output_dir, f"sample_{i:06d}.pt")
        torch.save(sample, out_path)

        if (i + 1) % 10 == 0 or i == 0 or (i + 1) == args.samples:
            print(
                f"[{i + 1:>6}/{args.samples}] saved {out_path} "
                f"(potential={potential_kind}, backend={args.backend})"
            )

    if args.backend == "gpu" and cp is not None:
        cp.cuda.Stream.null.synchronize()

    print("Done. Dataset written to", args.output_dir)


if __name__ == "__main__":
    main()
