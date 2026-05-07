#!/usr/bin/env python3
import argparse
import glob
import os
import time

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch

from train_tdse2d_delta_rollout import DeltaUNet2D, renormalize_psi


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize autoregressive 2D TDSE delta rollout vs sequence ground truth")
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--sample-index", type=int, default=0)
    p.add_argument(
        "--potential-preset",
        type=str,
        default="dataset",
        choices=["dataset", "center_square", "double_slit", "harmonic", "random_noise"],
        help="Potential shown/used in rollout. 'dataset' uses sample potential; others override it.",
    )
    p.add_argument("--potential-seed", type=int, default=1337, help="Seed for stochastic potential presets")
    p.add_argument("--potential-scale", type=float, default=1.0, help="Global scale multiplier for preset potential")
    p.add_argument("--rollout-steps", type=int, default=40)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--live", action="store_true", help="Show real-time interactive animation window")
    p.add_argument("--loop", action="store_true", help="Loop animation in live mode")
    p.add_argument("--infinite-live", action="store_true", help="In live mode, keep autoregressive rollout running until window is closed")
    p.add_argument(
        "--gt-mode",
        type=str,
        default="evolve",
        choices=["evolve", "cycle"],
        help="Ground-truth behavior in infinite-live mode: evolve with SOFT each frame, or cycle recorded sequence",
    )
    p.add_argument(
        "--speed-scale",
        type=float,
        default=1.0,
        help="Infinite-live speed multiplier. Example 5.0 makes evolution ~5x faster.",
    )
    p.add_argument(
        "--gt-solver-grid-size",
        type=int,
        default=0,
        help="If >0, run GT SOFT solver at this grid size (e.g. 80) and upsample back for display/error.",
    )
    p.add_argument("--out-gif", type=str, default="./tdse2d_delta_rollout.gif")
    p.add_argument("--no-save-gif", action="store_true")
    p.add_argument("--density-cmap", type=str, default="viridis", help="Colormap for density panels")
    p.add_argument("--error-cmap", type=str, default="magma", help="Colormap for error panel")
    p.add_argument("--motion-cmap", type=str, default="turbo", help="Colormap for motion panel")
    p.add_argument("--model-only", action="store_true", help="Disable GT comparison and render model-only rollout for maximum speed")
    p.add_argument("--no-error-motion", action="store_true", help="Disable error/motion panel updates for higher FPS")
    p.add_argument("--metrics-every", type=int, default=1, help="Compute relL2/GT-motion metrics every N frames")
    p.add_argument("--robust-lower-percentile", type=float, default=0.5, help="Lower percentile for density color scaling")
    p.add_argument("--robust-percentile", type=float, default=99.5, help="Upper percentile for density color scaling")
    p.add_argument("--density-gamma", type=float, default=0.6, help="Gamma for density display contrast (<1 brightens details)")
    return p.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_arg == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_sample(data_dir: str, sample_index: int):
    paths = sorted(glob.glob(os.path.join(data_dir, "sample_*.pt")))
    if not paths:
        raise RuntimeError(f"No sample_*.pt files found in {data_dir}")
    idx = max(0, min(sample_index, len(paths) - 1))
    sample = torch.load(paths[idx], map_location="cpu", weights_only=False)

    potential = sample["input"]["potential"].float()  # [N,N]
    psi_seq = sample["target"]["psi_seq"].float()  # [T+1,2,N,N]
    return potential, psi_seq, paths[idx]


def density(psi_ri: torch.Tensor) -> torch.Tensor:
    return psi_ri[0] ** 2 + psi_ri[1] ** 2


def robust_vmax(arr: np.ndarray, percentile: float) -> float:
    p = float(np.clip(percentile, 50.0, 100.0))
    v = float(np.percentile(arr, p))
    return max(v, 1e-12)


def normalize_density_display(arr: np.ndarray, low_p: float, high_p: float, gamma: float) -> np.ndarray:
    lp = float(np.clip(low_p, 0.0, 99.0))
    hp = float(np.clip(high_p, lp + 0.1, 100.0))
    lo = float(np.percentile(arr, lp))
    hi = float(np.percentile(arr, hp))
    if hi <= lo:
        hi = lo + 1e-12
    z = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    g = max(1e-3, float(gamma))
    return np.power(z, g)


def resize_real_field(field_2d: torch.Tensor, out_n: int) -> torch.Tensor:
    if int(field_2d.shape[-1]) == int(out_n):
        return field_2d
    t = field_2d.unsqueeze(0).unsqueeze(0)
    t = torch.nn.functional.interpolate(t, size=(out_n, out_n), mode="bilinear", align_corners=False)
    return t.squeeze(0).squeeze(0)


def resize_psi_ri(psi_ri: torch.Tensor, out_n: int, dx_new: float) -> torch.Tensor:
    if int(psi_ri.shape[-1]) == int(out_n):
        return psi_ri
    t = psi_ri.unsqueeze(0)
    t = torch.nn.functional.interpolate(t, size=(out_n, out_n), mode="bilinear", align_corners=False)
    t = renormalize_psi(t, dx_new)
    return t.squeeze(0)


def make_absorbing_mask_torch(n: int, length: float, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    dx = length / float(n)
    x1d = torch.arange(n, device=device, dtype=dtype) * dx - 0.5 * length
    x, y = torch.meshgrid(x1d, x1d, indexing="ij")
    edge = 0.5 * length
    absorb_start = 0.80 * edge
    absorb_strength = 1.5
    r = torch.maximum(x.abs(), y.abs())
    q = ((r - absorb_start) / (edge - absorb_start + 1e-8)).clamp(0.0, 1.0)
    return torch.exp(-absorb_strength * q * q)


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


def make_xy_grid_torch(n: int, length: float, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    dx = length / float(n)
    x1d = torch.arange(n, device=device, dtype=dtype) * dx - 0.5 * length
    return torch.meshgrid(x1d, x1d, indexing="ij")


def build_potential_preset(
    preset: str,
    n: int,
    length: float,
    scale: float,
    seed: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    x, y = make_xy_grid_torch(n=n, length=length, dtype=dtype, device=device)
    if preset == "center_square":
        half_w = 0.12 * length
        v = ((x.abs() < half_w) & (y.abs() < half_w)).to(dtype) * 14.0
        return scale * v

    if preset == "double_slit":
        barrier_h = 14.0
        barrier_w = 0.03 * length
        slit_w = 0.08 * length
        slit_sep = 0.18 * length
        wall = x.abs() < barrier_w
        slit_a = (y - 0.5 * slit_sep).abs() < slit_w
        slit_b = (y + 0.5 * slit_sep).abs() < slit_w
        v = torch.where(wall & (~(slit_a | slit_b)), torch.full_like(x, barrier_h), torch.zeros_like(x))
        return scale * v

    if preset == "harmonic":
        w = 0.18
        v = 0.5 * ((w * x) ** 2 + (w * y) ** 2)
        return scale * v

    if preset == "random_noise":
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))
        noise = torch.randn((n, n), generator=gen, device=device, dtype=dtype)
        k1d = 2.0 * np.pi * torch.fft.fftfreq(n, d=(length / n), device=device, dtype=dtype)
        k2 = k1d[:, None] ** 2 + k1d[None, :] ** 2
        corr_len = 0.10 * length
        filt = torch.exp(-0.5 * (corr_len ** 2) * k2)
        smooth = torch.fft.ifft2(torch.fft.fft2(noise.to(torch.complex64)) * filt.to(torch.complex64)).real.to(dtype)
        smooth = smooth - smooth.mean()
        smooth = smooth / smooth.std().clamp_min(1e-6)
        return scale * 2.0 * smooth

    return torch.zeros((n, n), device=device, dtype=dtype)


def build_gt_density_sequence_from_solver(
    psi0_ri: torch.Tensor,
    potential_real: torch.Tensor,
    seq_steps: int,
    dt: float,
    mass: float,
    substeps_per_step: int,
    normalize_every: int,
    dx: float,
    domain_length: float,
    gt_solver_grid_size: int,
    device: torch.device,
) -> list[np.ndarray]:
    n_native = int(potential_real.shape[-1])
    n_solver = int(gt_solver_grid_size) if int(gt_solver_grid_size) > 0 else n_native
    real_dtype = torch.float32
    complex_dtype = torch.complex64

    dx_solver = float(domain_length) / float(n_solver)

    potential_solver = resize_real_field(potential_real.to(device=device, dtype=real_dtype), n_solver)
    psi0_solver = resize_psi_ri(psi0_ri.to(device=device, dtype=real_dtype), n_solver, dx_solver)

    k1d = 2.0 * np.pi * torch.fft.fftfreq(n_solver, d=dx_solver, device=device, dtype=real_dtype)
    k_squared = k1d[:, None] ** 2 + k1d[None, :] ** 2
    absorbing_mask = make_absorbing_mask_torch(n=n_solver, length=domain_length, dtype=real_dtype, device=device).to(complex_dtype)

    gt_complex = (psi0_solver[0] + 1j * psi0_solver[1]).to(complex_dtype)
    potential_complex = potential_solver.to(device=device, dtype=complex_dtype)
    dt_sub = dt / max(1, substeps_per_step)

    rho_gt_all = []
    for t in range(seq_steps):
        for _ in range(max(1, substeps_per_step)):
            gt_complex = soft_step_torch(
                psi=gt_complex,
                potential=potential_complex,
                dt=dt_sub,
                mass=mass,
                k_squared=k_squared,
                absorbing_mask=absorbing_mask,
            )
        if normalize_every > 0 and ((t + 1) % normalize_every == 0):
            gt_prob = (gt_complex.abs() ** 2).sum() * (dx_solver ** 2)
            gt_complex = gt_complex / torch.sqrt(gt_prob.clamp_min(1e-12))

        rho_t = (gt_complex.abs() ** 2).to(real_dtype)
        if n_solver != n_native:
            rho_t = resize_real_field(rho_t, n_native)
        rho_gt_all.append(rho_t.detach().cpu().numpy())

    return rho_gt_all


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    potential, psi_seq, sample_path = load_sample(args.data_dir, args.sample_index)
    seq_steps = int(psi_seq.shape[0] - 1)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_args = ckpt.get("args", {})
    base_channels = int(model_args.get("base_channels", 32))
    renorm_each_step = bool(model_args.get("renorm_each_step", False))
    dx = float(ckpt.get("dataset_meta", {}).get("dx", 1.0 / max(1, int(potential.shape[-1]))))

    dataset_meta_path = os.path.join(args.data_dir, "dataset_meta.pt")
    dataset_meta = {}
    if os.path.exists(dataset_meta_path):
        dataset_meta = torch.load(dataset_meta_path, map_location="cpu", weights_only=False)

    dt = float(dataset_meta.get("dt", 0.02))
    mass = float(dataset_meta.get("mass", 1.0))
    domain_length = float(dataset_meta.get("domain_length", float(int(potential.shape[-1])) * dx))
    substeps_per_step = int(dataset_meta.get("substeps_per_step", 1))
    normalize_every = int(dataset_meta.get("normalize_every", 1))

    if args.potential_scale <= 0:
        raise RuntimeError("--potential-scale must be > 0")

    if args.potential_preset != "dataset":
        n = int(potential.shape[-1])
        potential = build_potential_preset(
            preset=args.potential_preset,
            n=n,
            length=domain_length,
            scale=args.potential_scale,
            seed=args.potential_seed,
            dtype=torch.float32,
            device=torch.device("cpu"),
        ).cpu()

    model = DeltaUNet2D(in_ch=3, out_ch=2, base=base_channels).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    steps = min(args.rollout_steps, seq_steps)
    if steps <= 0:
        raise RuntimeError("rollout-steps must be > 0 and sequence must contain at least 2 states")
    if args.infinite_live and (not args.live):
        raise RuntimeError("--infinite-live requires --live")
    if args.infinite_live and (not args.no_save_gif):
        raise RuntimeError("--infinite-live cannot be used while saving GIF. Add --no-save-gif.")
    if args.speed_scale <= 0:
        raise RuntimeError("--speed-scale must be > 0")
    if args.gt_solver_grid_size < 0:
        raise RuntimeError("--gt-solver-grid-size must be >= 0")
    if args.metrics_every <= 0:
        raise RuntimeError("--metrics-every must be > 0")

    psi0 = psi_seq[0].to(device)
    potential_b = potential.unsqueeze(0).to(device)

    preds = []
    with torch.no_grad():
        cur = psi0.unsqueeze(0)  # [1,2,N,N]
        for _ in range(steps):
            inp = torch.cat([potential_b.unsqueeze(1), cur], dim=1)  # [1,3,N,N]
            delta = model(inp)
            nxt = cur + delta
            if renorm_each_step:
                nxt = renormalize_psi(nxt, dx)
            preds.append(nxt.squeeze(0).detach().cpu())
            cur = nxt

    rho_potential = potential.numpy()
    rho_pred = [density(p).numpy() for p in preds]
    if args.model_only:
        rho_gt_all = [rho_pred[min(t, len(rho_pred) - 1)] for t in range(seq_steps)]
    elif args.potential_preset == "dataset":
        rho_gt_all = [density(psi_seq[t + 1]).numpy() for t in range(seq_steps)]
    else:
        rho_gt_all = build_gt_density_sequence_from_solver(
            psi0_ri=psi_seq[0],
            potential_real=potential,
            seq_steps=seq_steps,
            dt=dt,
            mass=mass,
            substeps_per_step=substeps_per_step,
            normalize_every=normalize_every,
            dx=dx,
            domain_length=domain_length,
            gt_solver_grid_size=args.gt_solver_grid_size,
            device=device,
        )
    rho_gt = rho_gt_all[:steps]
    model_grid_n = int(potential.shape[-1])

    rho_prev = density(psi_seq[0]).numpy()
    rho_motion = []
    if args.no_error_motion:
        rho_motion = [np.zeros_like(rho_pred[0]) for _ in range(len(rho_pred))]
    else:
        for r in rho_pred:
            rho_motion.append(np.abs(r - rho_prev))
            rho_prev = r

    gt0_disp = normalize_density_display(
        rho_gt[0], args.robust_lower_percentile, args.robust_percentile, args.density_gamma
    )
    pr0_disp = normalize_density_display(
        rho_pred[0], args.robust_lower_percentile, args.robust_percentile, args.density_gamma
    )

    fig, axs = plt.subplots(1, 5, figsize=(18, 4), constrained_layout=True)
    im0 = axs[0].imshow(rho_potential, cmap="coolwarm")
    axs[0].set_title(f"Potential V ({args.potential_preset})")
    plt.colorbar(im0, ax=axs[0], fraction=0.046)

    im1 = axs[1].imshow(gt0_disp, cmap=args.density_cmap, vmin=0.0, vmax=1.0)
    axs[1].set_title("Ground Truth disabled (model-only)" if args.model_only else "Ground Truth (SOFT) |psi|^2")

    im2 = axs[2].imshow(pr0_disp, cmap=args.density_cmap, vmin=0.0, vmax=1.0)
    axs[2].set_title(f"Model Rollout |psi|^2 (N={model_grid_n})")

    err0 = np.zeros_like(rho_pred[0]) if (args.no_error_motion or args.model_only) else np.abs(rho_pred[0] - rho_gt[0])
    im3 = axs[3].imshow(err0, cmap=args.error_cmap)
    if args.model_only:
        axs[3].set_title("Error panel disabled (model-only)")
    else:
        axs[3].set_title("|Model - Ground Truth| density" if not args.no_error_motion else "Error panel disabled")

    im4 = axs[4].imshow(rho_motion[0], cmap=args.motion_cmap)
    if args.model_only:
        axs[4].set_title("Motion panel disabled (model-only)")
    else:
        axs[4].set_title("|rho_t - rho_{t-1}|" if not args.no_error_motion else "Motion panel disabled")

    for a in axs:
        a.set_xticks([])
        a.set_yticks([])

    title = fig.suptitle("", fontsize=11)
    fps_state = {"last_t": None, "ema": 0.0}
    metric_state = {"rel": 0.0, "gt_mo": 0.0}

    def tick_fps() -> float:
        now = time.perf_counter()
        if fps_state["last_t"] is None:
            fps_state["last_t"] = now
            return 0.0
        dt_frame = max(1e-6, now - fps_state["last_t"])
        inst = 1.0 / dt_frame
        if fps_state["ema"] <= 0.0:
            fps_state["ema"] = inst
        else:
            fps_state["ema"] = 0.9 * fps_state["ema"] + 0.1 * inst
        fps_state["last_t"] = now
        return float(fps_state["ema"])

    def _update(frame_idx: int):
        gt = rho_gt[frame_idx]
        pr = rho_pred[frame_idx]
        if args.no_error_motion or args.model_only:
            er = err0
            mo = rho_motion[frame_idx]
        else:
            er = np.abs(pr - gt)
            mo = rho_motion[frame_idx]

        gt_disp = normalize_density_display(
            gt, args.robust_lower_percentile, args.robust_percentile, args.density_gamma
        )
        pr_disp = normalize_density_display(
            pr, args.robust_lower_percentile, args.robust_percentile, args.density_gamma
        )
        im1.set_data(gt_disp)
        im2.set_data(pr_disp)
        im3.set_data(er)
        im4.set_data(mo)

        if (not args.no_error_motion) and (not args.model_only):
            im3.set_clim(0.0, max(1e-8, float(er.max())))
            im4.set_clim(0.0, max(1e-10, float(mo.max())))

        if (not args.model_only) and (frame_idx % args.metrics_every == 0):
            metric_state["rel"] = float(np.linalg.norm(pr - gt) / max(1e-12, np.linalg.norm(gt)))
        rel = metric_state["rel"]
        fps_now = tick_fps()
        title.set_text(
            f"sample={os.path.basename(sample_path)} frame={frame_idx + 1}/{steps} "
            f"relL2(density)={rel:.4e} FPS={fps_now:.1f}"
        )
        if args.model_only:
            axs[1].set_title(f"Ground Truth disabled (model-only, t+{frame_idx + 1})")
        else:
            axs[1].set_title(f"Ground Truth (SOFT) |psi|^2 (t+{frame_idx + 1})")
        axs[2].set_title(f"Model Rollout |psi|^2 (N={model_grid_n}, t+{frame_idx + 1})")
        return im1, im2, im3, im4, title

    live_state = {
        "step": 0,
        "cur": psi0.unsqueeze(0).clone(),
        "prev_rho": density(psi_seq[0]).numpy(),
        "prev_gt": rho_gt_all[0],
    }

    n = int(potential.shape[-1])
    real_dtype = torch.float32
    complex_dtype = torch.complex64
    k1d = 2.0 * np.pi * torch.fft.fftfreq(n, d=dx, device=device, dtype=real_dtype)
    k_squared = k1d[:, None] ** 2 + k1d[None, :] ** 2
    absorbing_mask = make_absorbing_mask_torch(n=n, length=domain_length, dtype=real_dtype, device=device).to(complex_dtype)
    potential_complex = potential.to(device=device, dtype=complex_dtype)
    gt_solver_n = int(args.gt_solver_grid_size) if int(args.gt_solver_grid_size) > 0 else n
    dx_gt = float(domain_length) / float(gt_solver_n)

    potential_gt_real = resize_real_field(potential.to(device=device, dtype=real_dtype), gt_solver_n)
    psi0_gt_ri = resize_psi_ri(psi_seq[0].to(device=device, dtype=real_dtype), gt_solver_n, dx_gt)
    gt_complex = (psi0_gt_ri[0] + 1j * psi0_gt_ri[1]).to(complex_dtype)
    potential_gt_complex = potential_gt_real.to(dtype=complex_dtype)

    k1d_gt = 2.0 * np.pi * torch.fft.fftfreq(gt_solver_n, d=dx_gt, device=device, dtype=real_dtype)
    k_squared_gt = k1d_gt[:, None] ** 2 + k1d_gt[None, :] ** 2
    absorbing_mask_gt = make_absorbing_mask_torch(
        n=gt_solver_n, length=domain_length, dtype=real_dtype, device=device
    ).to(complex_dtype)

    model_substeps = max(1, int(round(args.speed_scale)))
    gt_dt_scale = float(args.speed_scale)

    def _update_infinite(_frame_idx: int):
        nonlocal gt_complex
        with torch.no_grad():
            nxt = live_state["cur"]
            for _ in range(model_substeps):
                inp = torch.cat([potential_b.unsqueeze(1), nxt], dim=1)
                delta = model(inp)
                nxt = nxt + delta
                if renorm_each_step:
                    nxt = renormalize_psi(nxt, dx)

        pr = density(nxt.squeeze(0).detach().cpu()).numpy()
        live_state["cur"] = nxt
        mo = np.abs(pr - live_state["prev_rho"])
        live_state["prev_rho"] = pr

        if args.model_only:
            gt = pr
            gt_idx = min(live_state["step"], seq_steps - 1)
        elif args.gt_mode == "cycle":
            gt_idx = live_state["step"] % seq_steps
            gt = rho_gt_all[gt_idx]
        else:
            dt_sub = (dt * gt_dt_scale) / max(1, substeps_per_step)
            for _ in range(max(1, substeps_per_step)):
                gt_complex_local = soft_step_torch(
                    psi=gt_complex,
                    potential=potential_gt_complex,
                    dt=dt_sub,
                    mass=mass,
                    k_squared=k_squared_gt,
                    absorbing_mask=absorbing_mask_gt,
                )
                gt_complex = gt_complex_local
            if normalize_every > 0 and ((live_state["step"] + 1) % normalize_every == 0):
                gt_prob = (gt_complex.abs() ** 2).sum() * (dx_gt ** 2)
                gt_complex = gt_complex / torch.sqrt(gt_prob.clamp_min(1e-12))
            gt_t = (gt_complex.abs() ** 2).to(real_dtype)
            if gt_solver_n != n:
                gt_t = resize_real_field(gt_t, n)
            gt = gt_t.detach().cpu().numpy()
            gt_idx = min(live_state["step"], seq_steps - 1)
        if args.no_error_motion or args.model_only:
            er = err0
            gt_mo = np.zeros_like(gt)
        else:
            er = np.abs(pr - gt)
            gt_mo = np.abs(gt - live_state["prev_gt"])
            live_state["prev_gt"] = gt

        gt_disp = normalize_density_display(
            gt, args.robust_lower_percentile, args.robust_percentile, args.density_gamma
        )
        pr_disp = normalize_density_display(
            pr, args.robust_lower_percentile, args.robust_percentile, args.density_gamma
        )
        im1.set_data(gt_disp)
        im2.set_data(pr_disp)
        im3.set_data(er)
        im4.set_data(mo)
        if (not args.no_error_motion) and (not args.model_only):
            im3.set_clim(0.0, max(1e-8, float(er.max())))
            im4.set_clim(0.0, max(1e-10, float(mo.max())))

        if (not args.model_only) and ((live_state["step"] % args.metrics_every) == 0):
            metric_state["rel"] = float(np.linalg.norm(pr - gt) / max(1e-12, np.linalg.norm(gt)))
            metric_state["gt_mo"] = float(gt_mo.mean()) if not args.no_error_motion else 0.0
        rel = metric_state["rel"]
        gt_mo_mean = metric_state["gt_mo"]
        step_num = live_state["step"] + 1
        fps_now = tick_fps()
        title.set_text(
            f"sample={os.path.basename(sample_path)} step={step_num} (infinite live) "
            f"relL2(density)={rel:.4e} GT_motion_mean={gt_mo_mean:.3e} "
            f"speed={args.speed_scale:.2f}x FPS={fps_now:.1f}"
        )
        if args.model_only:
            axs[1].set_title(f"Ground Truth disabled (model-only, step {step_num})")
        elif args.gt_mode == "cycle":
            axs[1].set_title(f"Ground Truth (SOFT, cycled) |psi|^2 (t+{(gt_idx + 1)})")
        else:
            axs[1].set_title(f"Ground Truth (SOFT, evolving@{gt_solver_n}) |psi|^2 (step {step_num})")
        axs[2].set_title(f"Model Rollout |psi|^2 (N={model_grid_n}, t+{step_num})")

        live_state["step"] += 1
        return im1, im2, im3, im4, title

    if args.infinite_live:
        ani = animation.FuncAnimation(
            fig,
            _update_infinite,
            frames=None,
            interval=1000 // max(1, args.fps),
            blit=False,
            repeat=False,
            cache_frame_data=False,
        )
    else:
        ani = animation.FuncAnimation(
            fig,
            _update,
            frames=steps,
            interval=1000 // max(1, args.fps),
            blit=False,
            repeat=args.loop,
        )

    if not args.no_save_gif:
        ani.save(args.out_gif, writer="pillow", fps=args.fps)
        print("Saved rollout animation to", args.out_gif)

    if args.live:
        if args.infinite_live:
            print("Showing infinite live rollout window. Close it to exit.")
        else:
            print("Showing live rollout window. Close it to exit.")
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    main()
