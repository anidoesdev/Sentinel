#!/usr/bin/env python
"""Generate SHAP and Integrated Gradients explanations for SENTINEL models.

Outputs:
    artifacts/explanations/shap_bar.png        — per-sensor SHAP importance (Gaussian)
    artifacts/explanations/ig_heatmap.png      — [timestep × sensor] IG map (VAE)
    Printed tables in terminal for both models.

Usage:
    python scripts/explain.py
    python scripts/explain.py --n-shap-windows 50 --n-steps 100
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from sentinel.data.cmapss import CMAPSSConfig, drop_low_variance, get_healthy_cycle, load_raw
from sentinel.explanations.ig_explainer import VAEIntegratedGradients
from sentinel.explanations.shap_explainer import GaussianSHAPExplainer
from sentinel.inference.baseline import GaussianAnomalyScorer
from sentinel.inference.vae_scorer import VAEAnomalyScorer
from sentinel.models.vae import VAE

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = Path("artifacts/vae/checkpoint.pt")
OUT_DIR = Path("artifacts/explanations")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SHAP + IG explanations for SENTINEL")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--fd-id", type=int, default=1)
    p.add_argument(
        "--n-shap-windows",
        type=int,
        default=30,
        help="Number of anomalous rows to explain with SHAP (more = slower)",
    )
    p.add_argument(
        "--n-steps",
        type=int,
        default=50,
        help="Riemann sum steps for Integrated Gradients",
    )
    p.add_argument(
        "--rul-threshold",
        type=int,
        default=30,
        help="RUL <= this value defines 'anomalous' in the test set",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_vae_scorer(ckpt_path: Path) -> tuple[VAEAnomalyScorer, VAE, list[str]]:
    ckpt = torch.load(ckpt_path, weights_only=False)
    model = VAE(**ckpt["model_config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    scorer = VAEAnomalyScorer(
        model=model,
        threshold=ckpt["threshold"],
        mean=pd.Series(ckpt["mean"]),
        std=pd.Series(ckpt["std"]),
        sensor_cols=ckpt["sensor_cols"],
        window_size=ckpt["model_config"]["window_size"],
        use_delta_features=ckpt.get("use_delta_features", False),
    )
    return scorer, model, ckpt["dropped_cols"]


def get_anomalous_windows(
    test_df: pd.DataFrame,
    rul_df: pd.DataFrame,
    vae_scorer: VAEAnomalyScorer,
    rul_threshold: int,
) -> np.ndarray:
    """Return normalised windows from engines with RUL <= rul_threshold."""
    anomalous_units = set(
        (i + 1) for i, rul in enumerate(rul_df["RUL"]) if rul <= rul_threshold
    )
    anom_df = test_df[test_df["unit"].isin(anomalous_units)]

    all_windows: list[np.ndarray] = []
    for _, group in anom_df.groupby("unit"):
        group_feat = vae_scorer._add_deltas(group)
        normed = VAEAnomalyScorer._normalize_df(
            group_feat, vae_scorer.mean, vae_scorer.std, vae_scorer.feature_cols
        )
        if len(normed) >= vae_scorer.window_size:
            windows = VAEAnomalyScorer._make_windows(
                normed, vae_scorer.feature_cols, vae_scorer.window_size
            )
            all_windows.append(windows)

    if not all_windows:
        raise ValueError("No anomalous windows found — check rul_threshold and data.")
    return np.concatenate(all_windows, axis=0)  # [N, window_size, n_sensors]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_shap_bar(
    importance_df: pd.DataFrame,
    out_path: Path,
    title: str = "Gaussian Scorer — Mean |SHAP| per Sensor",
) -> None:
    fig, ax = plt.subplots(figsize=(8, max(4, len(importance_df) * 0.4)))
    colors = ["#e74c3c" if i < 3 else "#95a5a6" for i in range(len(importance_df))]
    ax.barh(
        importance_df["sensor"][::-1],
        importance_df["mean_abs_shap"][::-1],
        color=colors[::-1],
    )
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(title)
    ax.axvline(0, color="black", linewidth=0.8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", out_path)


def plot_ig_heatmap(
    ig: np.ndarray,
    sensor_cols: list[str],
    out_path: Path,
    title: str = "VAE — Integrated Gradients (anomalous window)",
) -> None:
    # ig: [window_size, n_sensors]
    fig, ax = plt.subplots(figsize=(max(8, len(sensor_cols) * 0.6), 5))
    vmax = np.abs(ig).max()
    im = ax.imshow(
        ig.T,  # [n_sensors, window_size] — sensors on y-axis, time on x-axis
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
    )
    ax.set_yticks(range(len(sensor_cols)))
    ax.set_yticklabels(sensor_cols, fontsize=8)
    ax.set_xlabel("Timestep within window")
    ax.set_ylabel("Sensor")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Attribution (red=anomalous, blue=normal)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"No checkpoint at {args.checkpoint}. Run scripts/train_vae.py first."
        )

    config = CMAPSSConfig(data_dir=args.data_dir, fd_id=args.fd_id)

    # --- Load data ---
    logger.info("Loading CMAPSS data...")
    train_df = load_raw(config)
    train_df, dropped = drop_low_variance(train_df)
    healthy_df = get_healthy_cycle(train_df)

    test_df = load_raw(config, split="test")
    test_df = test_df.drop(columns=dropped)

    rul_df = pd.read_csv(
        args.data_dir / "raw" / f"RUL_FD00{args.fd_id}.txt",
        header=None, names=["RUL"],
    )

    # --- Load VAE scorer ---
    logger.info("Loading VAE checkpoint...")
    vae_scorer, vae_model, _ = load_vae_scorer(args.checkpoint)

    # =========================================================
    # Part 1: SHAP on Gaussian scorer
    # =========================================================
    logger.info("--- SHAP (Gaussian scorer) ---")
    gaussian = GaussianAnomalyScorer.fit(healthy_df)

    # Use raw (un-normalised) sensor readings for Gaussian SHAP — the scorer
    # computes z-scores internally, so it expects raw values.
    sensor_cols = gaussian.sensor_cols
    anom_mask = test_df["unit"].isin(
        set(i + 1 for i, r in enumerate(rul_df["RUL"]) if r <= args.rul_threshold)
    )
    anom_rows = test_df[anom_mask][sensor_cols].head(args.n_shap_windows)

    if len(anom_rows) == 0:
        logger.warning("No anomalous rows found for SHAP — skipping.")
    else:
        explainer = GaussianSHAPExplainer(
            scorer=gaussian,
            background_df=healthy_df,
            n_background=100,
        )
        logger.info("Computing SHAP values for %d rows...", len(anom_rows))
        shap_values = explainer.explain(anom_rows, n_samples=100)
        importance = explainer.feature_importance(shap_values)

        sep = "=" * 50
        print(f"\n{sep}")
        print("  SHAP — Gaussian Scorer: Top Sensors")
        print(sep)
        print(importance.head(10).to_string(index=False))
        print(sep)

        plot_shap_bar(importance, OUT_DIR / "shap_bar.png")

    # =========================================================
    # Part 2: Integrated Gradients on VAE
    # =========================================================
    logger.info("--- Integrated Gradients (VAE) ---")

    # Get healthy windows for the IG baseline (normalised, same as VAE training)
    healthy_windows_list: list[np.ndarray] = []
    for _, group in healthy_df.groupby("unit"):
        group_feat = vae_scorer._add_deltas(group)
        normed = VAEAnomalyScorer._normalize_df(
            group_feat, vae_scorer.mean, vae_scorer.std, vae_scorer.feature_cols
        )
        if len(normed) >= vae_scorer.window_size:
            w = VAEAnomalyScorer._make_windows(
                normed, vae_scorer.feature_cols, vae_scorer.window_size
            )
            healthy_windows_list.append(w)
    healthy_windows = np.concatenate(healthy_windows_list, axis=0)

    ig_explainer = VAEIntegratedGradients.from_healthy_windows(
        model=vae_model,
        windows=healthy_windows,
        n_steps=args.n_steps,
    )

    # Explain the most anomalous window from an anomalous engine
    anom_windows = get_anomalous_windows(test_df, rul_df, vae_scorer, args.rul_threshold)
    # Pick the window with the highest reconstruction error
    window_errors = vae_scorer._score_windows(anom_windows)
    worst_idx = int(np.argmax(window_errors))
    worst_window = anom_windows[worst_idx]  # [window_size, n_sensors]

    logger.info(
        "Explaining worst window (idx=%d, recon_error=%.4f)...",
        worst_idx, window_errors[worst_idx],
    )
    x_tensor = torch.from_numpy(worst_window).float().unsqueeze(0)  # [1, W, S]
    ig = ig_explainer.explain(x_tensor)

    sensor_importance = ig_explainer.sensor_importance(ig, vae_scorer.feature_cols)

    sep = "=" * 50
    print(f"\n{sep}")
    print("  Integrated Gradients — VAE: Top Sensors")
    print(f"  (worst window: recon_error={window_errors[worst_idx]:.4f})")
    print(sep)
    print(sensor_importance.head(10).to_string(index=False))
    print(sep)

    plot_ig_heatmap(ig, vae_scorer.feature_cols, OUT_DIR / "ig_heatmap.png")

    print(f"\nPlots saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
