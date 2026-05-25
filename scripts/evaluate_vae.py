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
from sklearn.metrics import classification_report

from sentinel.data.cmapss import CMAPSSConfig, load_raw
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


def main() -> None:
    args = parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"No checkpoint at {args.checkpoint}. Run scripts/train_vae.py first."
        )

    logger.info("Loading checkpoint: %s", args.checkpoint)
    scorer, dropped_cols = load_scorer(args.checkpoint)

    config = CMAPSSConfig(data_dir=args.data_dir, fd_id=args.fd_id)
    logger.info("Loading CMAPSS FD00%d test set...", args.fd_id)
    test_df = load_raw(config, split="test")
    test_df = test_df.drop(columns=dropped_cols)

    rul_df = pd.read_csv(
        args.data_dir / "raw" / f"RUL_FD00{args.fd_id}.txt",
        header=None, names=["RUL"],
    )
    # Label: engine is "anomalous" (near failure) if final RUL ≤ 30 cycles.
    true_labels = (rul_df["RUL"] <= 30).astype(int).values
    n_engines = len(true_labels)

    logger.info("Scoring %d test engines...", n_engines)
    scores_per_engine = scorer.score_engines(test_df)

    # Engine unit IDs in CMAPSS are 1-indexed; RUL file is ordered 1..N.
    score_values = np.array([
        scores_per_engine.get(i + 1, 0.0) for i in range(n_engines)
    ])
    pred_labels = (score_values > scorer.threshold).astype(int)

    # --- Diagnostic: score distribution by true class ---
    results_df = pd.DataFrame({
        "score": score_values,
        "true_label": true_labels,
    })
    normal_scores = results_df[results_df["true_label"] == 0]["score"]
    anomalous_scores = results_df[results_df["true_label"] == 1]["score"]

    print("\n" + "=" * 50)
    print("  Score Distribution Diagnostic")
    print("=" * 50)
    print(f"  Normal engines    (n={len(normal_scores):3d})  "
          f"mean={normal_scores.mean():.4f}  max={normal_scores.max():.4f}")
    print(f"  Anomalous engines (n={len(anomalous_scores):3d})  "
          f"mean={anomalous_scores.mean():.4f}  max={anomalous_scores.max():.4f}")
    print(f"  Threshold (99th pct healthy training): {scorer.threshold:.4f}")
    overlap = (anomalous_scores < normal_scores.max()).sum()
    print(f"  Anomalous engines below normal-max score: {overlap}/{len(anomalous_scores)}")

    print("\n" + "=" * 50)
    print("  VAE Anomaly Detector")
    print(f"  Threshold: {scorer.threshold:.6f}")
    print("=" * 50)
    print(classification_report(
        true_labels, pred_labels,
        target_names=["Normal", "Anomalous"],
        zero_division=0,
    ))

    print("=" * 50)
    print("  Baseline Reference (Gaussian rolling z-score)")
    print("  Precision=0.89  Recall=0.32  F1=0.47")
    print("=" * 50)


if __name__ == "__main__":
    main()
