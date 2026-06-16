#!/usr/bin/env python
"""Check for sensor data drift between healthy training data and test windows.

Simulates a production drift check: healthy training data is the reference,
a rolling window of test-set readings (from engines with low RUL) is the
current data. In a real system this would run on a schedule against a
TimescaleDB or Redis buffer of recent readings.

Usage:
    python scripts/monitor_drift.py
    python scripts/monitor_drift.py --window-size 500 --save-html
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from sentinel.data.cmapss import CMAPSSConfig, drop_low_variance, get_healthy_cycle, load_raw
from sentinel.monitoring.drift_monitor import DriftMonitor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sensor drift monitoring on CMAPSS")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--fd-id", type=int, default=1)
    p.add_argument(
        "--window-size",
        type=int,
        default=300,
        help="Number of recent test rows to treat as the 'current' production window",
    )
    p.add_argument(
        "--rul-threshold",
        type=int,
        default=30,
        help="Take the current window from engines with RUL <= this value",
    )
    p.add_argument(
        "--save-html",
        action="store_true",
        help="Save full interactive Evidently HTML report to artifacts/drift/report.html",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = CMAPSSConfig(data_dir=args.data_dir, fd_id=args.fd_id)

    # --- Reference: healthy training data ---
    logger.info("Loading healthy training data (reference)...")
    train_df = load_raw(config)
    train_df, dropped = drop_low_variance(train_df)
    healthy_df = get_healthy_cycle(train_df)

    # --- Current window: late-stage (near-failure) test readings ---
    logger.info("Loading test data (current window)...")
    test_df = load_raw(config, split="test")
    test_df = test_df.drop(columns=dropped)

    rul_df = pd.read_csv(
        args.data_dir / "raw" / f"RUL_FD00{args.fd_id}.txt",
        header=None, names=["RUL"],
    )
    anomalous_units = {i + 1 for i, r in enumerate(rul_df["RUL"]) if r <= args.rul_threshold}
    current_df = test_df[test_df["unit"].isin(anomalous_units)].tail(args.window_size)
    logger.info(
        "Current window: %d rows from %d near-failure engines.",
        len(current_df), len(anomalous_units),
    )

    # --- Fit monitor and run check ---
    monitor = DriftMonitor.fit(healthy_df)
    logger.info("Running drift check...")
    report = monitor.check(current_df)

    # --- Print results ---
    sep = "=" * 55
    print(f"\n{sep}")
    print(f"  Drift Report — CMAPSS FD00{args.fd_id}")
    print(f"  Reference: healthy training data")
    print(f"  Current:   {len(current_df)} rows (RUL <= {args.rul_threshold})")
    print(sep)
    status = "DRIFT DETECTED" if report.drift_detected else "No drift detected"
    print(f"  Status:  {status}")
    print(f"  Drifted: {report.n_drifted} / {report.n_sensors} sensors "
          f"({report.share_drifted:.0%})")

    if report.drifted_sensors:
        print(f"\n  Drifted sensors:")
        for sensor in report.drifted_sensors:
            details = report.sensor_details[sensor]
            p = details.get("p_value")
            p_str = f"{p:.4f}" if p is not None else "n/a"
            print(f"    {sensor:<20} p={p_str}")

    if report.sensor_details:
        print(f"\n  All sensors (KS p-value, lower = more drift):")
        sorted_sensors = sorted(
            report.sensor_details.items(),
            key=lambda kv: (kv[1].get("p_value") or 1.0),
        )
        for sensor, details in sorted_sensors[:10]:
            p = details.get("p_value")
            flag = " <-- DRIFT" if details.get("drift_detected") else ""
            p_str = f"{p:.4f}" if p is not None else "n/a"
            print(f"    {sensor:<20} p={p_str}{flag}")
    print(sep)

    # --- Optional HTML report ---
    if args.save_html:
        out = Path("artifacts/drift/report.html")
        monitor.save_html_report(current_df, out)
        print(f"\n  Interactive report: {out}")
        print(f"  Open in browser to see per-sensor distribution plots.")


if __name__ == "__main__":
    main()
