#!/usr/bin/env python
"""Evaluate the trained VAE against the Gaussian baseline on the CMAPSS test set.

Run after scripts/train_vae.py has produced artifacts/vae/checkpoint.pt.
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    roc_auc_score,
)

from sentinel.data.cmapss import CMAPSSConfig, get_healthy_cycle, load_raw
from sentinel.inference.vae_scorer import VAEAnomalyScorer
from sentinel.models.vae import VAE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = Path("artifacts/vae/checkpoint.pt")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate VAE vs Gaussian baseline")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--fd-id", type=int, default=1)
    p.add_argument("--threshold", type=float, default=None,
                   help="Override threshold directly (skips recalibration)")
    return p.parse_args()


def load_scorer(checkpoint_path: Path) -> tuple[VAEAnomalyScorer, list[str]]:
    """Reconstruct scorer from checkpoint. Returns (scorer, dropped_cols)."""
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


def recalibrate_threshold(
    scorer: VAEAnomalyScorer,
    train_df: pd.DataFrame,
    percentile: float = 99.0,
) -> float:
    """Recompute threshold using per-engine max scores on healthy training data.

    The original threshold was calibrated on per-WINDOW errors from early-life
    healthy cycles. This calibrates on per-ENGINE scores, which matches the
    scoring methodology used at inference — fixing the distribution mismatch.
    """
    healthy = get_healthy_cycle(train_df)
    scores = scorer.score_engines(healthy)
    return float(np.percentile(scores, percentile))


def main() -> None:
    args = parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"No checkpoint at {args.checkpoint}. Run scripts/train_vae.py first."
        )

    logger.info("Loading checkpoint: %s", args.checkpoint)
    scorer, dropped_cols = load_scorer(args.checkpoint)

    config = CMAPSSConfig(data_dir=args.data_dir, fd_id=args.fd_id)

    # --- Threshold: manual override, or recalibrate from training data ---
    if args.threshold is not None:
        logger.info("Using manual threshold override: %.4f", args.threshold)
        scorer.threshold = args.threshold
    else:
        logger.info("Recalibrating threshold on training data (per-engine)...")
        train_df = load_raw(config)
        train_df = train_df.drop(columns=dropped_cols)
        recalibrated_threshold = recalibrate_threshold(scorer, train_df, percentile=99.0)
        logger.info(
            "Threshold:  original=%.4f  recalibrated=%.4f",
            scorer.threshold, recalibrated_threshold,
        )
        scorer.threshold = recalibrated_threshold

    # --- Test set ---
    logger.info("Loading CMAPSS FD00%d test set...", args.fd_id)
    test_df = load_raw(config, split="test")
    test_df = test_df.drop(columns=dropped_cols)

    rul_df = pd.read_csv(
        args.data_dir / "raw" / f"RUL_FD00{args.fd_id}.txt",
        header=None, names=["RUL"],
    )
    true_labels = (rul_df["RUL"] <= 30).astype(int).values
    n_engines = len(true_labels)

    logger.info("Scoring %d test engines...", n_engines)
    scores_per_engine = scorer.score_engines(test_df)
    score_values = np.array([
        scores_per_engine.get(i + 1, 0.0) for i in range(n_engines)
    ])
    pred_labels = (score_values > scorer.threshold).astype(int)

    # --- Score distribution diagnostic ---
    results_df = pd.DataFrame({"score": score_values, "true_label": true_labels})
    normal_scores = results_df[results_df["true_label"] == 0]["score"]
    anomalous_scores = results_df[results_df["true_label"] == 1]["score"]

    print("\n" + "=" * 55)
    print("  Score Distribution")
    print("=" * 55)
    print(f"  Normal     (n={len(normal_scores):3d})  "
          f"mean={normal_scores.mean():.4f}  max={normal_scores.max():.4f}")
    print(f"  Anomalous  (n={len(anomalous_scores):3d})  "
          f"mean={anomalous_scores.mean():.4f}  max={anomalous_scores.max():.4f}")
    print(f"  Threshold (recalibrated, 99th pct per-engine): {scorer.threshold:.4f}")

    # --- AUROC: threshold-independent measure of discriminative power ---
    # AUROC answers: "what is the probability the model ranks a random anomalous
    # engine above a random normal engine?" 0.5 = random, 1.0 = perfect.
    auroc = roc_auc_score(true_labels, score_values)
    print(f"\n  AUROC: {auroc:.3f}  (0.5=random, 1.0=perfect)")

    # --- Precision-recall sweep: find the best achievable F1 ---
    # NOTE: finding the optimal threshold on the TEST SET is data leakage.
    # This is shown for diagnostic purposes only — in production, tune on a
    # held-out validation set.
    precisions, recalls, pr_thresholds = precision_recall_curve(true_labels, score_values)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_idx = int(np.argmax(f1_scores))
    best_f1 = f1_scores[best_idx]
    best_pr_threshold = float(pr_thresholds[min(best_idx, len(pr_thresholds) - 1)])
    print(f"  Best achievable F1: {best_f1:.2f}  "
          f"(P={precisions[best_idx]:.2f} R={recalls[best_idx]:.2f}) "
          f"at threshold={best_pr_threshold:.4f}")
    print("  (best-F1 threshold is oracle/data-leakage — use only for diagnosis)")

    # --- Classification report at recalibrated threshold ---
    print("\n" + "=" * 55)
    print("  VAE Anomaly Detector  (recalibrated threshold)")
    print(f"  Threshold: {scorer.threshold:.6f}")
    print("=" * 55)
    print(classification_report(
        true_labels, pred_labels,
        target_names=["Normal", "Anomalous"],
        zero_division=0,
    ))

    print("=" * 55)
    print("  Baseline Reference (Gaussian rolling z-score)")
    print("  Precision=0.89  Recall=0.32  F1=0.47")
    print("=" * 55)


if __name__ == "__main__":
    main()
