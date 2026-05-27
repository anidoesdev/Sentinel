#!/usr/bin/env python
"""Evaluate the VAE + Gaussian late-fusion ensemble on the CMAPSS test set.

No retraining required. Loads the VAE checkpoint, fits the Gaussian on training
data, fits the ensemble normalization, then evaluates on test.
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, precision_recall_curve, roc_auc_score

from sentinel.data.cmapss import CMAPSSConfig, drop_low_variance, get_healthy_cycle, load_raw
from sentinel.inference.baseline import GaussianAnomalyScorer
from sentinel.inference.ensemble_scorer import EnsembleScorer
from sentinel.inference.vae_scorer import VAEAnomalyScorer
from sentinel.models.vae import VAE

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = Path("artifacts/vae/checkpoint.pt")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VAE + Gaussian ensemble evaluation")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--fd-id", type=int, default=1)
    p.add_argument("--vae-weight", type=float, default=0.5,
                   help="Weight for VAE score in [0,1]; Gaussian gets 1-weight")
    p.add_argument("--threshold", type=float, default=None,
                   help="Override threshold (skips 99th-pct calibration)")
    return p.parse_args()


def load_vae_scorer(checkpoint_path: Path) -> tuple[VAEAnomalyScorer, list[str]]:
    ckpt = torch.load(checkpoint_path, weights_only=False)
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
    return scorer, ckpt["dropped_cols"]


def print_pr_analysis(true_labels: np.ndarray, score_values: np.ndarray) -> None:
    auroc = roc_auc_score(true_labels, score_values)
    precisions, recalls, pr_thresholds = precision_recall_curve(true_labels, score_values)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_idx = int(np.argmax(f1_scores))
    best_threshold = float(pr_thresholds[min(best_idx, len(pr_thresholds) - 1)])
    print(f"  AUROC: {auroc:.3f}  (0.5=random, 1.0=perfect)")
    print(f"  Best achievable F1: {f1_scores[best_idx]:.2f}  "
          f"(P={precisions[best_idx]:.2f} R={recalls[best_idx]:.2f})  "
          f"at threshold={best_threshold:.4f}  [oracle — diagnosis only]")


def main() -> None:
    args = parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"No checkpoint at {args.checkpoint}. Run scripts/train_vae.py first."
        )

    config = CMAPSSConfig(data_dir=args.data_dir, fd_id=args.fd_id)

    # --- Load VAE ---
    logger.info("Loading VAE checkpoint...")
    vae_scorer, dropped_cols = load_vae_scorer(args.checkpoint)

    # --- Fit Gaussian on healthy training data ---
    logger.info("Fitting Gaussian scorer on healthy training cycles...")
    train_df = load_raw(config)
    train_df, _ = drop_low_variance(train_df)
    healthy = get_healthy_cycle(train_df)
    gaussian_scorer = GaussianAnomalyScorer.fit(healthy)

    # --- Fit ensemble ---
    logger.info("Fitting ensemble normalization (vae_weight=%.1f)...", args.vae_weight)
    ensemble = EnsembleScorer.fit(
        vae_scorer=vae_scorer,
        gaussian_scorer=gaussian_scorer,
        healthy_df=healthy,
        percentile=99.0,
        vae_weight=args.vae_weight,
    )
    logger.info("Calibrated threshold (99th pct fused healthy score): %.4f", ensemble.threshold)

    if args.threshold is not None:
        ensemble.threshold = args.threshold
        logger.info("Overriding threshold to: %.4f", ensemble.threshold)

    # --- Test set ---
    logger.info("Loading test data and scoring engines...")
    test_df = load_raw(config, split="test")
    test_df = test_df.drop(columns=dropped_cols)

    rul_df = pd.read_csv(
        args.data_dir / "raw" / f"RUL_FD00{args.fd_id}.txt",
        header=None, names=["RUL"],
    )
    true_labels = (rul_df["RUL"] <= 30).astype(int).values
    n_engines = len(true_labels)

    scores_per_engine = ensemble.score_engines(test_df)
    score_values = np.array([scores_per_engine.get(i + 1, 0.0) for i in range(n_engines)])
    pred_labels = (score_values > ensemble.threshold).astype(int)

    # --- Results ---
    sep = "=" * 55
    print(f"\n{sep}")
    print(f"  Ensemble  (VAE {args.vae_weight:.0%} / Gaussian {1-args.vae_weight:.0%})")
    print(sep)
    print_pr_analysis(true_labels, score_values)
    print(f"  Threshold used: {ensemble.threshold:.4f}")
    print()
    print(classification_report(
        true_labels, pred_labels,
        target_names=["Normal", "Anomalous"],
        zero_division=0,
    ))

    print(sep)
    print("  Model Comparison")
    print(sep)
    print("  Gaussian baseline:  P=0.89  R=0.32  F1=0.47  AUROC≈0.67")
    print("  VAE (oracle thr):   P=0.38  R=0.88  F1=0.53  AUROC=0.75")
    print(f"  Ensemble:           see above")
    print(sep)


if __name__ == "__main__":
    main()
