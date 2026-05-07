#!/usr/bin/env python3
import argparse
import glob
import os
import time

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch

from train_tdse2d_delta_rollout import DeltaUNet2D, build_fourier_coord_features, renormalize_psi


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize model-only 2D TDSE delta rollout (single panel)")
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--sample-index", type=int, default=0)
    p.add_argument("--rollout-steps", type=int, default=120)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--live", action="store_true", help="Show interactive animation window")
    p.add_argument("--loop", action="store_true", help="Loop finite animation in live mode")
    p.add_argument("--infinite-live", action="store_true", help="Keep autoregressive rollout running until window close")
    p.add_argument("--speed-scale", type=float, default=1.0, help="Model rollout speed multiplier per rendered frame")
    p.add_argument("--density-cmap", type=str, default="turbo")
    p.add_argument("--robust-lower-percentile", type=float, default=1.0)
    p.add_argument("--robust-percentile", type=float, default=99.5)
    p.add_argument("--density-gamma", type=float, default=0.55)
    p.add_argument("--out-gif", type=str, default="./tdse2d_model_panel.gif")
    p.add_argument("--no-save-gif", action="store_true")
    return p.parse_args()


def resolve_device(device_arg: str) -> torch.device:
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


def load_sample(data_dir: str, sample_index: int):
    paths = sorted(glob.glob(os.path.join(data_dir, "sample_*.pt")))
    if not paths:
        raise RuntimeError(f"No sample_*.pt files found in {data_dir}")
    idx = max(0, min(sample_index, len(paths) - 1))
    sample = torch.load(paths[idx], map_location="cpu", weights_only=False)
    potential = sample["input"]["potential"].float()
    psi_seq = sample["target"]["psi_seq"].float()
    return potential, psi_seq, paths[idx]


def density(psi_ri: torch.Tensor) -> torch.Tensor:
    return psi_ri[0] ** 2 + psi_ri[1] ** 2


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


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    if args.speed_scale <= 0:
        raise RuntimeError("--speed-scale must be > 0")
    if args.infinite_live and (not args.live):
        raise RuntimeError("--infinite-live requires --live")
    if args.infinite_live and (not args.no_save_gif):
        raise RuntimeError("--infinite-live cannot be used while saving GIF. Add --no-save-gif")

    potential, psi_seq, sample_path = load_sample(args.data_dir, args.sample_index)
    seq_steps = int(psi_seq.shape[0] - 1)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_args = ckpt.get("args", {})
    base_channels = int(model_args.get("base_channels", 32))
    renorm_each_step = bool(model_args.get("renorm_each_step", False))
    fourier_base_freq = float(model_args.get("fourier_base_freq", np.pi))
    fourier_level_amp_decay = float(model_args.get("fourier_level_amp_decay", 1.0))
    dx = float(ckpt.get("dataset_meta", {}).get("dx", 1.0 / max(1, int(potential.shape[-1]))))

    state_dict = ckpt["model_state_dict"]
    model_in_ch = int(state_dict["enc1.net.0.weight"].shape[1])
    if model_in_ch < 3:
        raise RuntimeError(f"Invalid checkpoint input channels: {model_in_ch}")
    extra_in = model_in_ch - 3
    if extra_in == 0:
        fourier_levels = 0
    elif extra_in >= 2 and ((extra_in - 2) % 4 == 0):
        fourier_levels = int((extra_in - 2) // 4)
    else:
        raise RuntimeError(
            f"Unsupported checkpoint input channels={model_in_ch}; cannot infer Fourier feature layout"
        )

    model = DeltaUNet2D(in_ch=model_in_ch, out_ch=2, base=base_channels).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    psi0 = psi_seq[0].to(device)
    potential_b = potential.unsqueeze(0).to(device)
    coord_feats_1b = build_fourier_coord_features(
        batch_size=1,
        n=int(potential.shape[-1]),
        levels=fourier_levels,
        base_freq=fourier_base_freq,
        device=device,
        dtype=psi0.dtype,
        level_amp_decay=fourier_level_amp_decay,
    )
    model_grid_n = int(potential.shape[-1])

    steps = min(max(1, int(args.rollout_steps)), max(1, seq_steps))

    preds = []
    with torch.no_grad():
        cur = psi0.unsqueeze(0)
        for _ in range(steps):
            inp = torch.cat([potential_b.unsqueeze(1), cur], dim=1)
            if coord_feats_1b is not None:
                inp = torch.cat([inp, coord_feats_1b], dim=1)
            delta = model(inp)
            nxt = cur + delta
            if renorm_each_step:
                nxt = renormalize_psi(nxt, dx)
            preds.append(density(nxt.squeeze(0).detach().cpu()).numpy())
            cur = nxt

    first_disp = normalize_density_display(
        preds[0], args.robust_lower_percentile, args.robust_percentile, args.density_gamma
    )

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), constrained_layout=True)
    im = ax.imshow(first_disp, cmap=args.density_cmap, vmin=0.0, vmax=1.0)
    ax.set_xticks([])
    ax.set_yticks([])

    title = fig.suptitle("", fontsize=11)
    fps_state = {"last_t": None, "ema": 0.0}

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
        pr = preds[frame_idx]
        pr_disp = normalize_density_display(pr, args.robust_lower_percentile, args.robust_percentile, args.density_gamma)
        im.set_data(pr_disp)
        fps_now = tick_fps()
        step_num = frame_idx + 1
        ax.set_title(f"Model Rollout |psi|^2 (N={model_grid_n}, t+{step_num})")
        title.set_text(
            f"sample={os.path.basename(sample_path)} frame={step_num}/{steps} model-only speed={args.speed_scale:.2f}x FPS={fps_now:.1f}"
        )
        return im, title

    live_state = {"step": 0, "cur": psi0.unsqueeze(0).clone()}
    model_substeps = max(1, int(round(args.speed_scale)))

    def _update_infinite(_frame_idx: int):
        with torch.no_grad():
            nxt = live_state["cur"]
            for _ in range(model_substeps):
                inp = torch.cat([potential_b.unsqueeze(1), nxt], dim=1)
                if coord_feats_1b is not None:
                    inp = torch.cat([inp, coord_feats_1b], dim=1)
                delta = model(inp)
                nxt = nxt + delta
                if renorm_each_step:
                    nxt = renormalize_psi(nxt, dx)
        live_state["cur"] = nxt

        pr = density(nxt.squeeze(0).detach().cpu()).numpy()
        pr_disp = normalize_density_display(pr, args.robust_lower_percentile, args.robust_percentile, args.density_gamma)
        im.set_data(pr_disp)

        step_num = live_state["step"] + 1
        fps_now = tick_fps()
        ax.set_title(f"Model Rollout |psi|^2 (N={model_grid_n}, t+{step_num})")
        title.set_text(
            f"sample={os.path.basename(sample_path)} step={step_num} (model-only infinite) speed={args.speed_scale:.2f}x FPS={fps_now:.1f}"
        )

        live_state["step"] += 1
        return im, title

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
        print("Saved model-only rollout animation to", args.out_gif)

    if args.live:
        if args.infinite_live:
            print("Showing model-only infinite live rollout window. Close it to exit.")
        else:
            print("Showing model-only live rollout window. Close it to exit.")
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    main()
