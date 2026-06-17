"""Publish sensor readings to Redis Streams.

Usage:
    # Synthetic degradation replay (no data files needed):
    python scripts/stream_producer.py --mode synthetic --unit-id 1 --steps 150

    # Replay from a CMAPSS test CSV (requires data/raw/test_FD001.txt):
    python scripts/stream_producer.py --mode cmapss --unit-id 1 --interval 0.5
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

sys.path.insert(0, str(Path(__file__).parent.parent))

from sentinel.streaming.producer import SensorProducer


def main() -> None:
    parser = argparse.ArgumentParser(description="SENTINEL Redis Streams producer")
    parser.add_argument("--mode", choices=["synthetic", "cmapss"], default="synthetic")
    parser.add_argument("--unit-id", type=int, default=1)
    parser.add_argument("--steps", type=int, default=150, help="Steps for synthetic mode")
    parser.add_argument("--interval", type=float, default=0.5, help="Seconds between messages")
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    args = parser.parse_args()

    producer = SensorProducer(host=args.redis_host, port=args.redis_port)

    if args.mode == "synthetic":
        producer.replay_synthetic(
            unit_id=args.unit_id,
            total_steps=args.steps,
            interval_s=args.interval,
        )
    else:
        _replay_cmapss(producer, args.unit_id, args.interval)


def _replay_cmapss(producer: SensorProducer, unit_id: int, interval_s: float) -> None:
    import time
    import pandas as pd

    data_path = Path("data/raw/test_FD001.txt")
    if not data_path.exists():
        logging.error("CMAPSS test file not found at %s", data_path)
        sys.exit(1)

    sensor_cols = [
        "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8",
        "sensor_9", "sensor_11", "sensor_12", "sensor_13", "sensor_14",
        "sensor_15", "sensor_17", "sensor_20", "sensor_21",
    ]
    all_cols = (
        ["unit_id", "cycle", "op1", "op2", "op3"]
        + [f"s{i}" for i in range(1, 22)]
    )
    df = pd.read_csv(data_path, sep=r"\s+", header=None, names=all_cols)
    unit_df = df[df["unit_id"] == unit_id].reset_index(drop=True)

    keep = ["s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21"]
    logging.info("Replaying unit %d: %d cycles", unit_id, len(unit_df))

    for step, (_, row) in enumerate(unit_df.iterrows()):
        sensors = {name: float(row[raw]) for name, raw in zip(sensor_cols, keep)}
        msg_id = producer.publish(unit_id, sensors, step=step)
        logging.debug("step=%d msg=%s", step, msg_id)
        time.sleep(interval_s)

    logging.info("Replay complete.")


if __name__ == "__main__":
    main()
