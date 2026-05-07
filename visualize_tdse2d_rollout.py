#!/usr/bin/env python3
import argparse
import glob
import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch

from train_tdse2d_unet import Tdse2dUNet


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize autoregressive 2D TDSE rollout vs ground truth")
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--sample-index", type=int, default=0)
    p.add_argument("--rollout-steps", type=int, default=20)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out-gif", type=str, default="./tdse2d_rollout.gif")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--live", action="store_true", help="Show real-time interactive animation window")
    p.add_argument("--no-save-gif", action="store_true", help="Skip writing GIF to disk")
    p.add_argument("--loop", action="store_true", help="Loop animation in live mode")
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
    psi_t0 = sample["input"]["psi_t0"].float()  # [2,N,N]
    psi_gt = sample["target"]["psi_t"].float()  # [2,N,N]
    return potential, psi_t0, psi_gt, paths[idx]


def density(psi_ri: torch.Tensor) -> torch.Tensor:
    return psi_ri[0] ** 2 + psi_ri[1] ** 2


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    potential, psi_t0, psi_gt, sample_path = load_sample(args.data_dir, args.sample_index)
    n = int(potential.shape[-1])

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_args = ckpt.get("args", {})
    base_channels = int(model_args.get("base_channels", 32))

    model = Tdse2dUNet(in_ch=3, out_ch=2, base=base_channels).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    inp = torch.cat([potential.unsqueeze(0), psi_t0], dim=0).unsqueeze(0).to(device)  # [1,3,N,N]

    preds = []
    with torch.no_grad():
        cur = inp.clone()
        for _ in range(args.rollout_steps):
            pred = model(cur)
            preds.append(pred.detach().cpu().squeeze(0))
            cur = torch.cat([cur[:, :1], pred], dim=1)

    rho0 = density(psi_t0).numpy()
    rhogt = density(psi_gt).numpy()
    rho_preds = [density(p).numpy() for p in preds]

    vmin = 0.0
    vmax = float(max(np.max(rho0), np.max(rhogt), max(np.max(r) for r in rho_preds)))

    fig, axs = plt.subplots(1, 4, figsize=(15, 4), constrained_layout=True)
    im0 = axs[0].imshow(potential.numpy(), cmap="coolwarm")
    axs[0].set_title("Potential V")
    plt.colorbar(im0, ax=axs[0], fraction=0.046)

    im1 = axs[1].imshow(rho0, cmap="inferno", vmin=vmin, vmax=vmax)
    axs[1].set_title("|psi(t0)|^2")

    im2 = axs[2].imshow(rho_preds[0], cmap="inferno", vmin=vmin, vmax=vmax)
    axs[2].set_title("Pred density")

    err0 = np.abs(rho_preds[0] - rhogt)
    im3 = axs[3].imshow(err0, cmap="magma")
    axs[3].set_title("|Pred - GT| density")

    for a in axs:
        a.set_xticks([])
        a.set_yticks([])

    title = fig.suptitle("", fontsize=11)

    def _update(frame_idx: int):
        rho_p = rho_preds[frame_idx]
        err = np.abs(rho_p - rhogt)
        im2.set_data(rho_p)
        im3.set_data(err)
        im3.set_clim(0.0, max(1e-8, float(err.max())))

        rel = np.linalg.norm(rho_p - rhogt) / max(1e-12, np.linalg.norm(rhogt))
        title.set_text(
            f"sample={os.path.basename(sample_path)} frame={frame_idx + 1}/{len(rho_preds)} relL2(density)={rel:.4e}"
        )
        axs[2].set_title(f"Pred density (step {frame_idx + 1})")
        return im2, im3, title

    ani = animation.FuncAnimation(
        fig,
        _update,
        frames=len(rho_preds),
        interval=1000 // max(1, args.fps),
        blit=False,
        repeat=args.loop,
    )

    if not args.no_save_gif:
        ani.save(args.out_gif, writer="pillow", fps=args.fps)
        print("Saved rollout animation to", args.out_gif)

    if args.live:
        print("Showing live rollout window. Close it to exit.")
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    main()
