"""TimescaleDB writer for SENTINEL.

All writes use parameterised queries — no string interpolation.
Connection is created once and reused; reconnect on failure.

Tables (created by infra/init_db.sql):
  sensor_readings  — raw per-sensor values from the stream
  anomaly_scores   — per-inference scores for champion + challenger
  drift_events     — periodic drift check results
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Generator

import psycopg2
import psycopg2.extras
from psycopg2.extensions import connection as PgConnection

logger = logging.getLogger(__name__)

_DEFAULT_DSN = (
    "host=localhost port=5432 dbname=sentinel user=sentinel password=sentinel"
)


class TimescaleWriter:
    """Thin wrapper around psycopg2 for writing SENTINEL events to TimescaleDB."""

    def __init__(self, dsn: str = _DEFAULT_DSN) -> None:
        self.dsn = dsn
        self._conn: PgConnection | None = None

    def connect(self) -> None:
        self._conn = psycopg2.connect(self.dsn)
        self._conn.autocommit = False
        logger.info("TimescaleDB connected.")

    def close(self) -> None:
        if self._conn and not self._conn.closed:
            self._conn.close()

    @contextmanager
    def _cursor(self) -> Generator:
        if self._conn is None or self._conn.closed:
            self.connect()
        assert self._conn is not None
        cur = self._conn.cursor()
        try:
            yield cur
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cur.close()

    def write_sensor_readings(
        self,
        unit_id: int,
        sensors: dict[str, float],
        ts: datetime | None = None,
    ) -> None:
        """Insert one row per sensor for a single timestep."""
        ts = ts or datetime.now(timezone.utc)
        rows = [(ts, unit_id, name, value) for name, value in sensors.items()]
        with self._cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                "INSERT INTO sensor_readings (time, unit_id, sensor_name, value) VALUES %s",
                rows,
            )

    def write_anomaly_score(
        self,
        unit_id: int | None,
        model_version: str,
        modality: str,
        score: float,
        threshold: float,
        is_anomalous: bool,
        is_shadow: bool = False,
        ts: datetime | None = None,
    ) -> None:
        """Insert one anomaly score record."""
        ts = ts or datetime.now(timezone.utc)
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO anomaly_scores
                    (time, unit_id, model_version, modality, score, threshold, is_anomalous, is_shadow)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (ts, unit_id, model_version, modality, score, threshold, is_anomalous, is_shadow),
            )

    def write_drift_event(
        self,
        n_drifted: int,
        total: int,
        drifted_sensors: list[str],
        share: float,
        ts: datetime | None = None,
    ) -> None:
        ts = ts or datetime.now(timezone.utc)
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO drift_events
                    (time, n_drifted_sensors, total_sensors, drifted_sensors, share_drifted)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (ts, n_drifted, total, drifted_sensors, share),
            )

    def is_available(self) -> bool:
        """Return True if the database is reachable."""
        try:
            if self._conn is None or self._conn.closed:
                self.connect()
            assert self._conn is not None
            self._conn.cursor().execute("SELECT 1")
            return True
        except Exception:
            return False
