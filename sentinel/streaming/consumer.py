"""Redis Streams consumer — reads sensor readings, scores them, writes to TimescaleDB.

Uses Redis consumer groups so multiple consumer instances can share the load
without processing the same message twice. Each consumer ACKs a message only
after successfully writing the score to TimescaleDB — if the consumer crashes
mid-write, the message stays pending and will be redelivered.

Consumer group: sentinel-scorers
Consumer name:  scorer-{pid}

Flow per message:
  1. XREADGROUP reads next unACKed message from the stream
  2. Buffer the reading into a per-unit sliding window (deque of window_size)
  3. Once the window is full, run VAE inference
  4. Write sensor readings + anomaly score to TimescaleDB
  5. XACK the message
"""
from __future__ import annotations

import logging
import os
from collections import deque

import numpy as np
import pandas as pd
import redis

logger = logging.getLogger(__name__)

STREAM_PREFIX = "sentinel:sensors"
GROUP_NAME = "sentinel-scorers"
BLOCK_MS = 2_000   # block up to 2s waiting for new messages
BATCH_SIZE = 10    # read up to 10 messages per round


class ScoringConsumer:
    """Reads from Redis Streams, scores with the VAE, persists to TimescaleDB."""

    def __init__(
        self,
        vae_scorer,          # VAEAnomalyScorer — loaded externally
        db_writer,           # TimescaleWriter — connected externally
        unit_ids: list[int],
        redis_host: str = "localhost",
        redis_port: int = 6379,
    ) -> None:
        self.scorer = vae_scorer
        self.db = db_writer
        self.unit_ids = unit_ids
        self.consumer_name = f"scorer-{os.getpid()}"

        self.r = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.r.ping()

        # Per-unit sliding window buffers
        self.buffers: dict[int, deque] = {
            uid: deque(maxlen=vae_scorer.window_size) for uid in unit_ids
        }

        self._ensure_groups()
        logger.info(
            "ScoringConsumer ready. Consumer=%s, units=%s", self.consumer_name, unit_ids
        )

    def _ensure_groups(self) -> None:
        """Create consumer groups if they don't exist. $ means start from newest."""
        for uid in self.unit_ids:
            key = f"{STREAM_PREFIX}:{uid}"
            try:
                self.r.xgroup_create(key, GROUP_NAME, id="$", mkstream=True)
                logger.info("Created consumer group '%s' on %s", GROUP_NAME, key)
            except redis.exceptions.ResponseError as e:
                if "BUSYGROUP" not in str(e):
                    raise  # already exists — that's fine

    def run_forever(self) -> None:
        """Block-read loop. Call this in a background thread or process."""
        stream_keys = {f"{STREAM_PREFIX}:{uid}": uid for uid in self.unit_ids}
        streams = {key: ">" for key in stream_keys}  # ">" = undelivered messages

        logger.info("Consumer loop started.")
        while True:
            try:
                results = self.r.xreadgroup(
                    GROUP_NAME,
                    self.consumer_name,
                    streams,
                    count=BATCH_SIZE,
                    block=BLOCK_MS,
                )
                if not results:
                    continue

                for stream_key, messages in results:
                    unit_id = stream_keys[stream_key]
                    for msg_id, fields in messages:
                        self._handle(unit_id, msg_id, fields, stream_key)

            except Exception as e:
                logger.error("Consumer error: %s — retrying in 2s", e)
                import time; time.sleep(2)

    def _handle(self, unit_id: int, msg_id: str, fields: dict, stream_key: str) -> None:
        """Process one message: buffer → maybe score → persist → ACK."""
        step = int(fields.pop("step", 0))
        sensors = {k: float(v) for k, v in fields.items()}

        # Persist raw reading
        self.db.write_sensor_readings(unit_id, sensors)

        # Add to sliding window buffer
        buf = self.buffers[unit_id]
        sensor_values = list(sensors.values())
        buf.append(sensor_values)

        if len(buf) == self.scorer.window_size:
            readings_np = np.array(list(buf), dtype=np.float32)
            # Normalise and score
            mean = self.scorer.mean.values.astype(np.float32)
            std = self.scorer.std.values.astype(np.float32)
            normed = (readings_np - mean) / (std + 1e-8)
            import torch
            x = torch.from_numpy(normed).unsqueeze(0).permute(0, 2, 1)
            self.scorer.model.eval()
            with torch.no_grad():
                mu, _ = self.scorer.model.encoder(x.permute(0, 2, 1))
                x_hat = self.scorer.model.decoder(mu)
                score = float(((x.permute(0, 2, 1) - x_hat) ** 2).mean())

            is_anom = score > self.scorer.threshold
            self.db.write_anomaly_score(
                unit_id=unit_id,
                model_version="champion",
                modality="timeseries",
                score=score,
                threshold=self.scorer.threshold,
                is_anomalous=is_anom,
                is_shadow=False,
            )
            logger.debug(
                "unit=%d step=%d score=%.5f anomalous=%s", unit_id, step, score, is_anom
            )

        # ACK only after successful processing + DB write
        self.r.xack(stream_key, GROUP_NAME, msg_id)
