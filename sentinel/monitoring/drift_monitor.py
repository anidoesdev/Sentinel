"""Sensor data drift monitoring using Evidently.

Drift monitoring is distinct from anomaly detection:
- Anomaly detection: is this individual reading abnormal?
- Drift monitoring: has the *distribution* of recent readings shifted
  from what the model was trained on?

A fleet-wide sensor calibration drift won't trigger individual anomaly
alerts but will degrade model performance silently over time. Drift
monitoring catches this.

Evidently uses the Kolmogorov-Smirnov test per sensor (continuous data).
KS statistic measures the maximum difference between two CDFs — p < 0.05
means the distributions are statistically different.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from evidently.legacy.metric_preset import DataDriftPreset
from evidently.legacy.metrics import DataDriftTable, DatasetDriftMetric
from evidently.legacy.pipeline.column_mapping import ColumnMapping
from evidently.legacy.report import Report

logger = logging.getLogger(__name__)


@dataclass
class DriftReport:
    drift_detected: bool
    n_drifted: int
    n_sensors: int
    share_drifted: float
    drifted_sensors: list[str]
    sensor_details: dict       # {sensor: {drift_detected, p_value}}
    raw: dict = field(repr=False)


class DriftMonitor:
    """Fits on healthy reference data and checks incoming windows for drift.

    The reference dataset is the healthy training distribution — what the
    models were trained to expect. The current dataset is a recent window
    of production readings (e.g. last 5 minutes from a Redis buffer).
    """

    def __init__(
        self,
        reference_df: pd.DataFrame,
        sensor_cols: list[str],
        drift_threshold: float = 0.05,
    ) -> None:
        self.reference_df = reference_df[sensor_cols].copy()
        self.sensor_cols = sensor_cols
        self.drift_threshold = drift_threshold
        logger.info(
            "DriftMonitor ready — reference: %d rows, %d sensors.",
            len(self.reference_df), len(sensor_cols),
        )

    @classmethod
    def fit(
        cls,
        healthy_df: pd.DataFrame,
        sensor_cols: list[str] | None = None,
        drift_threshold: float = 0.05,
    ) -> "DriftMonitor":
        if sensor_cols is None:
            sensor_cols = [c for c in healthy_df.columns if c.startswith("sensor_")]
        if not sensor_cols:
            raise ValueError("No sensor columns found. Pass sensor_cols explicitly.")
        return cls(healthy_df, sensor_cols, drift_threshold)

    def check(self, current_df: pd.DataFrame, min_rows: int = 30) -> DriftReport:
        """Check whether current_df has drifted from the reference distribution.

        Args:
            current_df: recent production readings — same schema as reference.
            min_rows:   minimum rows for a reliable KS test.
        """
        current = current_df[self.sensor_cols].copy()

        if len(current) < min_rows:
            logger.warning(
                "Only %d rows in current window (need >= %d for reliable KS test).",
                len(current), min_rows,
            )

        column_mapping = ColumnMapping(numerical_features=self.sensor_cols)
        report = Report(metrics=[
            DatasetDriftMetric(drift_share_threshold=0.5),
            DataDriftTable(),
        ])
        report.run(
            reference_data=self.reference_df,
            current_data=current,
            column_mapping=column_mapping,
        )
        return self._parse(report.as_dict())

    def _parse(self, result: dict) -> DriftReport:
        metrics = result.get("metrics", [])

        dataset_result = next(
            (m["result"] for m in metrics if m["metric"] == "DatasetDriftMetric"), {}
        )
        drift_detected = dataset_result.get("dataset_drift", False)
        share_drifted = dataset_result.get("share_of_drifted_columns", 0.0)
        n_drifted = dataset_result.get("number_of_drifted_columns", 0)

        table_result = next(
            (m["result"] for m in metrics if m["metric"] == "DataDriftTable"), {}
        )
        drift_by_col = table_result.get("drift_by_columns", {})

        drifted_sensors = [c for c, s in drift_by_col.items() if s.get("drift_detected")]
        sensor_details = {
            col: {
                "drift_detected": stats.get("drift_detected"),
                "p_value": stats.get("drift_score"),
            }
            for col, stats in drift_by_col.items()
        }

        return DriftReport(
            drift_detected=drift_detected,
            n_drifted=n_drifted,
            n_sensors=len(self.sensor_cols),
            share_drifted=share_drifted,
            drifted_sensors=drifted_sensors,
            sensor_details=sensor_details,
            raw=result,
        )

    def save_html_report(self, current_df: pd.DataFrame, out_path: Path) -> None:
        """Save a full interactive Evidently HTML report (open in browser)."""
        current = current_df[self.sensor_cols].copy()
        column_mapping = ColumnMapping(numerical_features=self.sensor_cols)
        report = Report(metrics=[DataDriftPreset()])
        report.run(
            reference_data=self.reference_df,
            current_data=current,
            column_mapping=column_mapping,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report.save_html(str(out_path))
        logger.info("HTML drift report saved: %s", out_path)
