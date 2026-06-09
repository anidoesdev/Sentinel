"""Pydantic schemas for all SENTINEL serving endpoints.

Keeping schemas in a separate module makes them importable by tests and
by client code without pulling in the full FastAPI app.
"""
from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field, field_validator


class SensorWindow(BaseModel):
    """One window of sensor readings for the time-series scorer.

    readings: list of N timesteps, each a list of sensor values.
              Expected shape: [window_size, n_sensors] — typically [30, 14].
    unit_id:  machine unit identifier (for tracking in the dashboard).
    """

    unit_id: int = Field(..., ge=1, description="Machine unit ID (1-indexed)")
    readings: list[list[float]] = Field(
        ...,
        description="Sensor window: [timesteps, sensors]. Must match trained window_size.",
    )

    @field_validator("readings")
    @classmethod
    def check_non_empty(cls, v: list[list[float]]) -> list[list[float]]:
        if not v or not v[0]:
            raise ValueError("readings must be a non-empty 2D list")
        return v


class AnomalyScore(BaseModel):
    """Anomaly detection result returned by all scoring endpoints."""

    unit_id: int | None = None
    anomaly_score: float = Field(..., description="Raw anomaly score (higher = more anomalous)")
    is_anomalous: bool = Field(..., description="True if score exceeds the fitted threshold")
    threshold: float = Field(..., description="Threshold in use at inference time")
    modality: str = Field(..., description="'timeseries' or 'audio'")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of the inference",
    )


class HealthResponse(BaseModel):
    status: str  # "ok" or "degraded"
    models_loaded: list[str]
    models_missing: list[str]
