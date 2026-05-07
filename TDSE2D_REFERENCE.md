# 2D TDSE Pipeline Reference (Dataset → Train → Visualize)

This document summarizes the current 2D TDSE workflow in this repo and gives copy-paste commands for future runs.

## What each script does

- `tdse2d_sequence_generator.py`
  - Generates `.pt` sequence samples using a 2D SOFT solver.
  - Each sample stores:
    - input: `potential`, `psi_t0`
    - target: `psi_seq` (shape `[T+1, 2, N, N]`)
  - Writes `dataset_meta.pt` in the output directory.

- `train_tdse2d_delta_rollout.py`
  - Trains a delta-step UNet (`psi_next = psi_cur + delta`) with autoregressive rollout loss.
  - Supports rollout curriculum and horizon-weighting.
  - Saves best checkpoint as `best_tdse2d_delta.pt`.

- `visualize_tdse2d_delta_rollout.py`
  - Full comparison visualizer (potential, GT, model, error, motion).
  - Supports infinite live rollout, GT modes (`cycle`/`evolve`), lower-res GT solver, and performance flags.

- `visualize_tdse2d_model_panel.py`
  - Lightweight single-panel model-only visualizer for max runtime speed.

---

## Environment

From repo root:

```bash
source .venv/bin/activate
```

GPU/MPS notes:
- For training/visualization speed on Mac, pass `--device mps` where available.
- The model-panel visualizer defaults to `--device auto`.

---

## 1) Generate a dataset

## A. Mixed potentials (default blend)

```bash
python tdse2d_sequence_generator.py \
  --output-dir ./temp/tdse2d_seq_96_long \
  --samples 2400 \
  --grid-size 96 \
  --domain-length 24.0 \
  --dt 0.02 \
  --sequence-steps 80 \
  --substeps-per-step 2 \
  --mass 1.0 \
  --normalize-every 1 \
  --dtype float32 \
  --backend torch \
  --device auto \
  --potential-mode mixed \
  --seed 1337
```

## B. Double-slit-only dataset (focused training)

```bash
python tdse2d_sequence_generator.py \
  --output-dir ./temp/tdse2d_seq_256_doubleslit \
  --samples 180 \
  --grid-size 256 \
  --domain-length 24.0 \
  --dt 0.02 \
  --sequence-steps 40 \
  --substeps-per-step 2 \
  --mass 1.0 \
  --normalize-every 1 \
  --dtype float32 \
  --backend torch \
  --device auto \
  --potential-mode double_slit \
  --seed 1337
```

`--potential-mode` choices:
- `mixed`
- `double_slit`
- `harmonic`
- `random_noise`

---

## 2) Train delta-rollout model

## A. Mid-size 96 run (balanced)

```bash
python train_tdse2d_delta_rollout.py \
  --data-dir ./temp/tdse2d_seq_96_long \
  --output-dir ./temp/tdse2d_delta_96_mid \
  --device mps \
  --batch-size 12 \
  --epochs 50 \
  --lr 8e-4 \
  --weight-decay 1e-6 \
  --max-samples 1200 \
  --base-channels 24 \
  --rollout-steps 24 \
  --rollout-curriculum-start 4 \
  --rollout-curriculum-epochs 25 \
  --horizon-loss-power 2.0 \
  --physics-loss-weight 0.15 \
  --renorm-each-step \
  --num-workers 0
```

## B. 256 double-slit-focused run

```bash
python train_tdse2d_delta_rollout.py \
  --data-dir ./temp/tdse2d_seq_256_doubleslit \
  --output-dir ./temp/tdse2d_delta_256_doubleslit \
  --device mps \
  --batch-size 2 \
  --epochs 60 \
  --lr 6e-4 \
  --weight-decay 1e-6 \
  --base-channels 16 \
  --rollout-steps 16 \
  --rollout-curriculum-start 4 \
  --rollout-curriculum-epochs 30 \
  --horizon-loss-power 2.0 \
  --physics-loss-weight 0.15 \
  --renorm-each-step \
  --num-workers 0
```

Key training knobs:
- `--rollout-steps`: final horizon.
- `--rollout-curriculum-start` + `--rollout-curriculum-epochs`: horizon ramp.
- `--horizon-loss-power`: emphasizes late-step error when > 1.
- `--physics-loss-weight`: norm conservation penalty weight.

---

## 3) Visualize (full comparison)

```bash
python visualize_tdse2d_delta_rollout.py \
  --data-dir ./temp/tdse2d_seq_256_doubleslit \
  --checkpoint ./temp/tdse2d_delta_256_doubleslit/best_tdse2d_delta.pt \
  --sample-index 3 \
  --device mps \
  --live --infinite-live --gt-mode evolve \
  --speed-scale 6 \
  --gt-solver-grid-size 80 \
  --no-error-motion \
  --metrics-every 10 \
  --no-save-gif
```

Useful options:
- `--potential-preset {dataset,center_square,double_slit,harmonic,random_noise}`
- `--gt-mode {cycle,evolve}`
- `--gt-solver-grid-size 80` (run GT solver cheaper than model grid)
- `--model-only` (disables GT comparison logic)
- `--no-error-motion` (skip heavy panel updates)

---

## 4) Visualize (single-panel, fastest)

```bash
python visualize_tdse2d_model_panel.py \
  --data-dir ./temp/tdse2d_seq_256_doubleslit \
  --checkpoint ./temp/tdse2d_delta_256_doubleslit/best_tdse2d_delta.pt \
  --sample-index 3 \
  --live --infinite-live \
  --speed-scale 1 \
  --no-save-gif
```

This script is intended for high-FPS model-only playback.

---

## Performance tips

- Always set `--device mps` (or `auto` where supported).
- In full visualizer, biggest speed hits are:
  - GT evolve FFT path
  - high `--speed-scale`
  - error/motion panel updates
- For speed:
  - use `--model-only`
  - use `--no-error-motion`
  - reduce `--speed-scale`
  - reduce GT solver grid via `--gt-solver-grid-size`

---

## Common issues

- **No space left on device** when saving checkpoint:
  - Free disk space under `./temp` and rerun.

- **Slow visualization (few FPS):**
  - Ensure `--device mps`.
  - Switch to `visualize_tdse2d_model_panel.py` for fastest mode.

- **GT looks reset/cycled unexpectedly:**
  - Check `--gt-mode`; use `evolve` for continuous solver propagation.

---

## Suggested workflow

1. Generate dataset (`tdse2d_sequence_generator.py`)
2. Train (`train_tdse2d_delta_rollout.py`)
3. Validate visually with full viewer (`visualize_tdse2d_delta_rollout.py`)
4. Use single-panel model viewer (`visualize_tdse2d_model_panel.py`) for fast interactive playback
