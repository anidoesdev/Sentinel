"""Redis Streams producer — publishes sensor readings to a stream.

In production this would be the edge collector reading from physical sensors
(OPC-UA, MQTT, or direct ADC). Here it replays CMAPSS test data at a
configurable rate to simulate real-time ingestion.

Redis Streams vs Kafka:
  Redis Streams is a persistent, append-only log built into Redis. It supports
  consumer groups (multiple consumers reading from the same stream), message
  acknowledgement, and replay from any offset — the same core guarantees as
  Kafka, with no separate broker. For a single-machine deployment it's simpler
  and uses the Redis instance we already have. At high throughput (>100k msg/s)
  or multi-datacenter replication, Kafka wins.

Stream key: sentinel:sensors:{unit_id}
Message fields: one per sensor + "step" counter
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import redis

logger = logging.getLogger(__name__)

STREAM_PREFIX = "sentinel:sensors"
STREAM_MAXLEN = 10_000  # keep last 10k messages per unit (FIFO trim)


class SensorProducer:
    """Publishes sensor readings to Redis Streams.

    Each call to `publish` adds one message to the stream for the given unit.
    The consumer reads from these streams and runs model inference.
    """

    def __init__(self, host: str = "localhost", port: int = 6379) -> None:
        self.r = redis.Redis(host=host, port=port, decode_responses=True)
        self.r.ping()
        logger.info("SensorProducer connected to Redis %s:%d", host, port)

    def publish(self, unit_id: int, sensors: dict[str, float], step: int = 0) -> str:
        """Publish one reading. Returns the Redis stream message ID."""
        stream_key = f"{STREAM_PREFIX}:{unit_id}"
        fields = {"step": str(step), **{k: str(v) for k, v in sensors.items()}}
        msg_id = self.r.xadd(stream_key, fields, maxlen=STREAM_MAXLEN, approximate=True)
        return msg_id

    def replay_synthetic(
        self,
        unit_id: int = 1,
        total_steps: int = 150,
        interval_s: float = 0.5,
        sensor_names: list[str] | None = None,
    ) -> None:
        """Replay a synthetic degradation sequence (no data file needed).

        Phases: healthy (0-40) → degrading (40-90) → anomalous (90+).
        Mirrors the /ws/replay endpoint so the stream and the WebSocket
        produce consistent data for integration testing.
        """
        if sensor_names is None:
            sensor_names = [
                "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8",
                "sensor_9", "sensor_11", "sensor_12", "sensor_13", "sensor_14",
                "sensor_15", "sensor_17", "sensor_20", "sensor_21",
            ]
        degrading = {"sensor_3", "sensor_4", "sensor_7", "sensor_11", "sensor_12"}
        rng = np.random.default_rng(42)

        logger.info("Starting synthetic replay: unit=%d, steps=%d", unit_id, total_steps)
        for step in range(total_steps):
            degradation = max(0.0, min(2.0, (step - 40) / 50.0))
            reading = rng.normal(0.0, 1.0, len(sensor_names)).tolist()
            for i, name in enumerate(sensor_names):
                if name in degrading:
                    reading[i] += degradation * rng.normal(1.5, 0.3)

            sensors = {name: round(reading[i], 4) for i, name in enumerate(sensor_names)}
            msg_id = self.publish(unit_id, sensors, step)
            logger.debug("Published step %d → %s", step, msg_id)
            time.sleep(interval_s)

        logger.info("Replay complete.")
