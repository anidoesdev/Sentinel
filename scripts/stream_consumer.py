"""Consume sensor readings from Redis Streams and score with the VAE model.

Usage:
    python scripts/stream_consumer.py --unit-ids 1 2 3

Requires:
    - Redis running (docker compose up redis)
    - TimescaleDB running (docker compose up timescaledb)
    - Trained VAE checkpoint at artifacts/vae/checkpoint.pt

This is the Kafka/Redis Streams ingestion layer: a long-running process that
reads from sentinel:sensors:{unit_id} streams using consumer groups, scores
each buffered window with the champion VAE, and writes results to TimescaleDB.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="SENTINEL Redis Streams scoring consumer")
    parser.add_argument("--unit-ids", type=int, nargs="+", default=[1])
    parser.add_argument("--checkpoint", default="artifacts/vae/checkpoint.pt")
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--db-host", default="localhost")
    parser.add_argument("--db-port", type=int, default=5432)
    args = parser.parse_args()

    # Load VAE scorer
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        logger.error("VAE checkpoint not found at %s", ckpt_path)
        sys.exit(1)

    import pandas as pd
    import torch
    from sentinel.models.vae import VAE
    from sentinel.inference.vae_scorer import VAEAnomalyScorer

    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
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
    logger.info("VAE champion loaded (threshold=%.5f)", scorer.threshold)

    # Connect TimescaleDB
    from sentinel.storage.timescale import TimescaleWriter
    db = TimescaleWriter(
        host=args.db_host,
        port=args.db_port,
    )
    if db.is_available():
        logger.info("TimescaleDB connected.")
    else:
        logger.warning("TimescaleDB unavailable — scores will NOT be persisted.")

    # Start consumer loop
    from sentinel.streaming.consumer import ScoringConsumer
    consumer = ScoringConsumer(
        vae_scorer=scorer,
        db_writer=db,
        unit_ids=args.unit_ids,
        redis_host=args.redis_host,
        redis_port=args.redis_port,
    )
    logger.info("Starting consumer loop (Ctrl+C to stop)…")
    consumer.run_forever()


if __name__ == "__main__":
    main()
