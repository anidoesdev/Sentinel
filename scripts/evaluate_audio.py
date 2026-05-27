#!/usr/bin/env python
"""Evaluate the AST-based audio anomaly scorer on the MIMII dataset.

Workflow:
  1. Load MIMII fan (or other machine type) audio files
  2. Fit AudioAnomalyScorer on normal training audio (downloads AST on first run)
  3. Score the test set (held-out normals + all abnormals)
  4. Report AUROC, best-F1 (oracle), and classification report at calibrated threshold

Usage:
    python scripts/evaluate_audio.py
    python scripts/evaluate_audio.py --machine-type pump --snr-db 0
    python scripts/evaluate_audio.py --device cuda  # if GPU available

MIMII setup (one-time):
    Download 6dB_fan.zip from https://zenodo.org/record/3678171
    Extract so that:
        data/mimii/6dB_fan/id_00/normal/*.wav
        data/mimii/6dB_fan/id_00/abnormal/*.wav
"""
import argparse
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, precision_recall_curve, roc_auc_score
from torch.utils.data import DataLoader

from sentinel.data.mimii import MIMIIConfig, make_datasets
from sentinel.inference.audio_scorer import AudioAnomalyScorer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AST audio anomaly detection on MIMII")
    p.add_argument("--data-dir", type=Path, default=Path("data/mimii"))
    p.add_argument(
        "--machine-type", default="pump", choices=["fan", "pump", "slider", "valve"]
    )
    p.add_argument(
        "--machine-id",
        default="id_00",
        help="Filter to one machine ID, e.g. 'id_00'. Default: id_00.",
    )
    p.add_argument("--n-pca", type=int, default=32, help="PCA components (default 32)")
    p.add_argument(
        "--threshold-pct",
        type=float,
        default=99.0,
        help="Percentile of normal training scores used to set threshold (default 99). "
             "Lower this (e.g. 90) to increase anomaly recall at the cost of more false positives.",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override the calibrated threshold directly (e.g. 61.4). "
             "Skips percentile calibration.",
    )
    p.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device for AST inference",
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument(
        "--save-scorer",
        type=Path,
        default=None,
        help="If set, save the fitted scorer to this path (e.g. artifacts/audio/scorer.pkl)",
    )
    return p.parse_args()


def print_pr_analysis(true_labels: np.ndarray, scores: np.ndarray) -> None:
    auroc = roc_auc_score(true_labels, scores)
    precisions, recalls, pr_thresholds = precision_recall_curve(true_labels, scores)
    f1 = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_idx = int(np.argmax(f1))
    best_thr = float(pr_thresholds[min(best_idx, len(pr_thresholds) - 1)])
    print(f"  AUROC:        {auroc:.3f}  (0.5=random, 1.0=perfect)")
    print(
        f"  Best F1 (oracle threshold): {f1[best_idx]:.2f}  "
        f"(P={precisions[best_idx]:.2f}  R={recalls[best_idx]:.2f})  "
        f"at score={best_thr:.1f}  [oracle — use for diagnosis only]"
    )


def main() -> None:
    args = parse_args()

    config = MIMIIConfig(
        data_dir=args.data_dir,
        machine_type=args.machine_type,
        machine_id=args.machine_id,
        batch_size=args.batch_size,
    )

    logger.info("Building datasets for %s %s...", args.machine_type, args.machine_id)
    train_ds, test_ds = make_datasets(config)

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers
    )

    logger.info("Fitting AudioAnomalyScorer (AST downloads ~330MB on first run)...")
    scorer = AudioAnomalyScorer.fit(
        train_loader,
        device=args.device,
        n_pca_components=args.n_pca,
        threshold_percentile=args.threshold_pct,
    )

    if args.threshold is not None:
        logger.info("Overriding threshold: %.2f → %.2f", scorer.threshold, args.threshold)
        scorer.threshold = args.threshold

    if args.save_scorer is not None:
        scorer.save(args.save_scorer)

    logger.info("Scoring test set...")
    scores, true_labels = scorer.score(test_loader)
    preds = (scores > scorer.threshold).astype(int)

    sep = "=" * 58
    print(f"\n{sep}")
    print(f"  Audio Anomaly Detection — MIMII {args.machine_type} {args.machine_id}")
    print(f"  Threshold: {scorer.threshold:.2f}  (pct={args.threshold_pct if args.threshold is None else 'overridden'})")
    print(sep)
    print_pr_analysis(true_labels, scores)
    print()
    print(
        classification_report(
            true_labels,
            preds,
            target_names=["Normal", "Anomalous"],
            zero_division=0,
        )
    )
    print(sep)
    print("  Sentinel model comparison (CMAPSS time series):")
    print("  Gaussian baseline:  P=0.89  R=0.32  F1=0.47")
    print("  VAE (oracle):       P=0.38  R=0.88  F1=0.53  AUROC=0.75")
    print("  Ensemble:           see evaluate_ensemble.py")
    print(f"  Audio (this):       see above  [different dataset — MIMII]")
    print(sep)


if __name__ == "__main__":
    main()
