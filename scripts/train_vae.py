#!/usr/bin/env python
"""Train the VAE anomaly detector on CMAPSS and save an inference checkpoint.

Checkpoint format (artifacts/vae/checkpoint.pt):
    state_dict          — model weights
    model_config        — {input_dim, latent_dim, window_size}
    sensor_cols         — base sensor column names (before delta expansion)
    mean / std          — normalization parameters for all feature_cols
    threshold           — 99th-percentile healthy-window reconstruction error
    dropped_cols        — sensors removed by variance filter
    use_delta_features  — whether delta features were used in training
"""
import argparse
import logging
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch

from sentinel.data.cmapss import (
    CMAPSSConfig,
    add_delta_features,
    drop_low_variance,
    get_healthy_cycle,
    load_raw,
    make_windows,
    normalize,
)
from sentinel.inference.vae_scorer import VAEAnomalyScorer
from sentinel.training.train_vae import TrainConfig, train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path("artifacts/vae")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train VAE anomaly detector on CMAPSS")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--fd-id", type=int, default=1, choices=[1, 2, 3, 4])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--latent-dim", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-delta-features", action="store_true",
                   help="Disable delta features (use raw sensor values only)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    use_delta = not args.no_delta_features
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    config = CMAPSSConfig(data_dir=args.data_dir, fd_id=args.fd_id)

    logger.info("Loading training data...")
    df = load_raw(config)
    df, dropped = drop_low_variance(df)
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    logger.info("Sensors: %d (dropped %d: %s)", len(sensor_cols), len(dropped), dropped)

    healthy = get_healthy_cycle(df)

    if use_delta:
        # Add per-cycle first-difference features to the healthy training set.
        # Delta features capture gradual degradation trends that are nearly
        # invisible in absolute sensor values.
        healthy, delta_cols = add_delta_features(healthy, sensor_cols)
        feature_cols = sensor_cols + delta_cols
        logger.info("Delta features enabled: input_dim %d → %d", len(sensor_cols), len(feature_cols))
    else:
        feature_cols = sensor_cols

    # Normalize AFTER computing deltas so delta statistics (mean≈0, small std)
    # are captured correctly and not distorted by the raw-value scale.
    healthy_normed, mean, std = normalize(healthy, feature_cols)
    windows = make_windows(healthy_normed, feature_cols)
    logger.info("Training windows: %s  input_dim=%d", windows.shape, len(feature_cols))

    cfg = TrainConfig(
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )

    mlflow.set_experiment("sentinel-vae")
    with mlflow.start_run():
        mlflow.log_params({
            "fd_id": args.fd_id,
            "epochs": cfg.epochs,
            "latent_dim": cfg.latent_dim,
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "seed": cfg.seed,
            "n_windows": windows.shape[0],
            "input_dim": len(feature_cols),
            "use_delta_features": use_delta,
        })

        model = train(windows, input_dim=len(feature_cols), cfg=cfg)

        logger.info("Calibrating anomaly threshold...")
        # Pass raw (un-normalized) healthy data; scorer normalizes internally.
        healthy_raw = get_healthy_cycle(load_raw(config).drop(columns=dropped))
        scorer = VAEAnomalyScorer.fit(
            model=model,
            healthy_df=healthy_raw,
            mean=mean,
            std=std,
            sensor_cols=sensor_cols,
            window_size=cfg.window_size,
            use_delta_features=use_delta,
        )
        logger.info("Threshold (99th pct healthy error): %.6f", scorer.threshold)
        mlflow.log_metric("anomaly_threshold", scorer.threshold)

        checkpoint = {
            "state_dict": model.state_dict(),
            "model_config": {
                "input_dim": len(feature_cols),
                "latent_dim": cfg.latent_dim,
                "window_size": cfg.window_size,
            },
            "sensor_cols": sensor_cols,
            "mean": mean.to_dict(),
            "std": std.to_dict(),
            "threshold": scorer.threshold,
            "dropped_cols": dropped,
            "use_delta_features": use_delta,
        }
        checkpoint_path = ARTIFACTS_DIR / "checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)
        mlflow.log_artifact(str(checkpoint_path))
        logger.info("Checkpoint saved → %s", checkpoint_path)


if __name__ == "__main__":
    main()
