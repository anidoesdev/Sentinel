#!/usr/bin/env python
"""Train the VAE anomaly detector on CMAPSS and save an inference checkpoint.

Checkpoint format (artifacts/vae/checkpoint.pt):
    state_dict    — model weights
    model_config  — {input_dim, latent_dim, window_size}
    sensor_cols   — list of sensor column names used
    mean / std    — normalization parameters (dicts) for inference
    threshold     — 99th-percentile healthy-window reconstruction error
    dropped_cols  — sensors removed by variance filter
"""
import argparse
import logging
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch

from sentinel.data.cmapss import (
    CMAPSSConfig,
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
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Data ---
    config = CMAPSSConfig(data_dir=args.data_dir, fd_id=args.fd_id)

    logger.info("Loading training data...")
    df = load_raw(config)
    df, dropped = drop_low_variance(df)
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    logger.info("Sensors: %d (dropped %d: %s)", len(sensor_cols), len(dropped), dropped)

    # get_healthy_cycle returns raw data; we normalize separately so we can
    # store mean/std in the checkpoint for use at inference time.
    healthy = get_healthy_cycle(df)
    healthy_normed, mean, std = normalize(healthy, sensor_cols)
    windows = make_windows(healthy_normed, sensor_cols)
    logger.info("Training windows: %s", windows.shape)

    # --- Training ---
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
            "n_sensors": len(sensor_cols),
        })

        # train() logs per-epoch loss to the active run automatically.
        model = train(windows, input_dim=len(sensor_cols), cfg=cfg)

        # Fit scorer on raw (un-normalized) healthy data; VAEAnomalyScorer
        # normalizes internally using the stored mean/std.
        logger.info("Calibrating anomaly threshold...")
        scorer = VAEAnomalyScorer.fit(
            model=model,
            healthy_df=healthy,
            mean=mean,
            std=std,
            sensor_cols=sensor_cols,
            window_size=cfg.window_size,
        )
        logger.info("Threshold (99th pct healthy error): %.6f", scorer.threshold)
        mlflow.log_metric("anomaly_threshold", scorer.threshold)

        # --- Save checkpoint ---
        checkpoint = {
            "state_dict": model.state_dict(),
            "model_config": {
                "input_dim": len(sensor_cols),
                "latent_dim": cfg.latent_dim,
                "window_size": cfg.window_size,
            },
            "sensor_cols": sensor_cols,
            "mean": mean.to_dict(),
            "std": std.to_dict(),
            "threshold": scorer.threshold,
            "dropped_cols": dropped,
        }
        checkpoint_path = ARTIFACTS_DIR / "checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)
        mlflow.log_artifact(str(checkpoint_path))
        logger.info("Checkpoint saved → %s", checkpoint_path)


if __name__ == "__main__":
    main()
