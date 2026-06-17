"""SENTINEL FastAPI inference server.

Routes:
  GET  /health                   — readiness check, lists loaded models
  POST /score/timeseries         — score one sensor window (JSON body)
  POST /score/audio              — score one WAV file (multipart upload)
  WS   /ws/timeseries/{unit_id}  — streaming: send JSON readings, receive scores

Run with:
    python scripts/serve.py
    # or directly:
    uvicorn sentinel.serving.app:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import asyncio
import io
import logging
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from sentinel.serving import model_store
from sentinel.serving.schemas import AnomalyScore, HealthResponse, SensorWindow
from sentinel.storage.timescale import TimescaleWriter

_db: TimescaleWriter = TimescaleWriter()  # connects lazily; no crash if DB is down

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lifespan: load models once at startup, clean up on shutdown
# ---------------------------------------------------------------------------

_load_results: dict[str, bool] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _load_results
    logger.info("Loading SENTINEL models...")
    _load_results = model_store.load_all()
    loaded = [k for k, v in _load_results.items() if v]
    missing = [k for k, v in _load_results.items() if not v]
    logger.info("Models ready: %s  |  Missing: %s", loaded or "none", missing or "none")
    yield
    logger.info("Shutting down SENTINEL serving.")


app = FastAPI(
    title="SENTINEL Anomaly Detection",
    description="Real-time multimodal anomaly detection for industrial machines.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Returns loaded/missing model list. 200 = at least one model ready."""
    loaded = [k for k, v in _load_results.items() if v]
    missing = [k for k, v in _load_results.items() if not v]
    return HealthResponse(
        status="ok" if loaded else "degraded",
        models_loaded=loaded,
        models_missing=missing,
    )


# ---------------------------------------------------------------------------
# POST /score/timeseries
# ---------------------------------------------------------------------------

@app.post("/score/timeseries", response_model=AnomalyScore)
async def score_timeseries(window: SensorWindow) -> AnomalyScore:
    """Score one window of sensor readings using the VAE anomaly scorer.

    Body example:
        {"unit_id": 1, "readings": [[641.82, 0.02, ...], ...]}  (30 × 14)

    Runs champion inference synchronously (returned to client) and challenger
    inference in the background (logged to TimescaleDB, not returned).
    Inference is CPU-bound, so both run in thread-pool executors.
    """
    scorer = model_store.get_vae_scorer()
    if scorer is None:
        raise HTTPException(status_code=503, detail="VAE scorer not loaded")

    readings_np = np.array(window.readings, dtype=np.float32)
    loop = asyncio.get_event_loop()

    # Champion inference — returned to client
    score = await loop.run_in_executor(None, _score_ts_window, scorer, readings_np)
    is_anomalous = bool(score > scorer.threshold)

    # Persist champion score (fire-and-forget, no crash if DB is down)
    asyncio.create_task(_write_score_async(
        unit_id=window.unit_id or 0,
        model_version="champion",
        modality="timeseries",
        score=score,
        threshold=scorer.threshold,
        is_anomalous=is_anomalous,
        is_shadow=False,
    ))

    # Shadow inference — challenger model, logged only
    challenger = model_store.get_vae_challenger()
    if challenger is not None:
        asyncio.create_task(_run_shadow_async(
            challenger=challenger,
            readings_np=readings_np,
            unit_id=window.unit_id or 0,
        ))

    return AnomalyScore(
        unit_id=window.unit_id,
        anomaly_score=float(score),
        is_anomalous=is_anomalous,
        threshold=float(scorer.threshold),
        modality="timeseries",
    )


async def _write_score_async(
    unit_id: int, model_version: str, modality: str,
    score: float, threshold: float, is_anomalous: bool, is_shadow: bool,
) -> None:
    """Write one anomaly score to TimescaleDB — non-blocking, swallows errors."""
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: _db.write_anomaly_score(
                unit_id, model_version, modality, score, threshold, is_anomalous, is_shadow
            ),
        )
    except Exception as exc:
        logger.debug("TimescaleDB write skipped (%s)", exc)


async def _run_shadow_async(challenger, readings_np: np.ndarray, unit_id: int) -> None:
    """Run challenger inference and log to TimescaleDB (shadow deployment)."""
    try:
        loop = asyncio.get_event_loop()
        shadow_score = await loop.run_in_executor(None, _score_ts_window, challenger, readings_np)
        await _write_score_async(
            unit_id=unit_id,
            model_version="challenger",
            modality="timeseries",
            score=shadow_score,
            threshold=challenger.threshold,
            is_anomalous=bool(shadow_score > challenger.threshold),
            is_shadow=True,
        )
    except Exception as exc:
        logger.debug("Shadow inference skipped (%s)", exc)


def _score_ts_window(scorer, readings_np: np.ndarray) -> float:
    """Blocking inference — called inside run_in_executor."""
    window_size = scorer.window_size
    n_sensors = len(scorer.sensor_cols)

    if readings_np.shape != (window_size, n_sensors):
        raise ValueError(
            f"Expected readings shape ({window_size}, {n_sensors}), "
            f"got {readings_np.shape}"
        )

    # Normalize using training mean/std
    mean = scorer.mean.values.astype(np.float32)
    std = scorer.std.values.astype(np.float32)
    normalized = (readings_np - mean) / (std + 1e-8)

    x = torch.from_numpy(normalized).unsqueeze(0)  # [1, window_size, n_sensors]
    x = x.permute(0, 2, 1)                          # [1, n_sensors, window_size] for Conv1d

    scorer.model.eval()
    with torch.no_grad():
        x_hat, _, _ = scorer.model(x)
        error = torch.mean((x - x_hat) ** 2).item()
    return error


# ---------------------------------------------------------------------------
# POST /score/audio
# ---------------------------------------------------------------------------

@app.post("/score/audio", response_model=AnomalyScore)
async def score_audio(file: UploadFile = File(...)) -> AnomalyScore:
    """Score one WAV file using the AST-based audio anomaly scorer.

    Accepts a multipart file upload. The WAV must be at 16kHz (or will be
    resampled). Returns the squared Mahalanobis distance from the normal cluster.
    """
    scorer = model_store.get_audio_scorer()
    if scorer is None:
        raise HTTPException(status_code=503, detail="Audio scorer not loaded")

    raw = await file.read()
    loop = asyncio.get_event_loop()
    score = await loop.run_in_executor(None, _score_audio_bytes, scorer, raw)

    is_anomalous = bool(score > scorer.threshold)
    asyncio.create_task(_write_score_async(
        unit_id=0,
        model_version="ast-v1",
        modality="audio",
        score=score,
        threshold=scorer.threshold,
        is_anomalous=is_anomalous,
        is_shadow=False,
    ))

    return AnomalyScore(
        unit_id=None,
        anomaly_score=float(score),
        is_anomalous=is_anomalous,
        threshold=float(scorer.threshold),
        modality="audio",
    )


def _score_audio_bytes(scorer, raw_bytes: bytes) -> float:
    """Blocking audio inference — runs the full AST pipeline on one clip."""
    import torchaudio.functional as F_audio

    data, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(data.T)  # [channels, samples]

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.squeeze(0)  # [samples]

    target_sr = scorer.encoder.sample_rate
    if sr != target_sr:
        waveform = F_audio.resample(waveform, sr, target_sr)

    # Pad or trim to 10 seconds
    n_samples = target_sr * 10
    if waveform.shape[0] < n_samples:
        waveform = torch.nn.functional.pad(waveform, (0, n_samples - waveform.shape[0]))
    else:
        waveform = waveform[:n_samples]

    embedding = scorer.encoder.encode_batch(waveform.unsqueeze(0))  # [1, 768]
    pca_proj = scorer.pca.transform(embedding)
    scaled = scorer.scaler.transform(pca_proj)
    return float(np.sum(scaled ** 2))


# ---------------------------------------------------------------------------
# WS /ws/timeseries/{unit_id}
# ---------------------------------------------------------------------------

# Per-unit sliding window buffers: unit_id → deque of sensor reading lists
_unit_buffers: dict[int, deque] = {}


@app.websocket("/ws/timeseries/{unit_id}")
async def ws_timeseries(websocket: WebSocket, unit_id: int) -> None:
    """Streaming anomaly scoring over WebSocket.

    Client sends JSON messages: {"sensors": [v1, v2, ..., v14]}
    Server responds with JSON: {"anomaly_score": 0.12, "is_anomalous": false, "buffered": 15}

    The server maintains a sliding window buffer per unit. Once the buffer
    reaches window_size readings, it scores on every new incoming reading
    (sliding window, step=1). Before the buffer is full, it returns the
    current buffer length so the client knows how many more readings are needed.
    """
    scorer = model_store.get_vae_scorer()
    if scorer is None:
        await websocket.close(code=1013, reason="VAE scorer not loaded")
        return

    await websocket.accept()
    window_size = scorer.window_size
    n_sensors = len(scorer.sensor_cols)

    if unit_id not in _unit_buffers:
        _unit_buffers[unit_id] = deque(maxlen=window_size)
    buf = _unit_buffers[unit_id]

    logger.info("WebSocket connected: unit_id=%d", unit_id)

    try:
        while True:
            data = await websocket.receive_json()
            sensors = data.get("sensors")

            if not sensors or len(sensors) != n_sensors:
                await websocket.send_json({
                    "error": f"Expected {n_sensors} sensor values, got {len(sensors) if sensors else 0}"
                })
                continue

            buf.append(sensors)

            if len(buf) < window_size:
                await websocket.send_json({"buffered": len(buf), "window_size": window_size})
                continue

            # Full window available — score in thread pool
            readings_np = np.array(list(buf), dtype=np.float32)
            loop = asyncio.get_event_loop()
            score = await loop.run_in_executor(None, _score_ts_window, scorer, readings_np)

            await websocket.send_json({
                "unit_id": unit_id,
                "anomaly_score": round(score, 6),
                "is_anomalous": bool(score > scorer.threshold),
                "threshold": round(scorer.threshold, 6),
            })

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: unit_id=%d", unit_id)
    except Exception as e:
        logger.error("WebSocket error for unit %d: %s", unit_id, e)
        await websocket.close(code=1011)


# ---------------------------------------------------------------------------
# WS /ws/replay  — synthetic degradation demo (no live sensors required)
# ---------------------------------------------------------------------------

# CMAPSS FD001 sensor names (14 kept after dropping low-variance columns)
_SENSOR_NAMES = [
    "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8",
    "sensor_9", "sensor_11", "sensor_12", "sensor_13", "sensor_14",
    "sensor_15", "sensor_17", "sensor_20", "sensor_21",
]

# Which sensors degrade in FD001 (HPC degradation fault mode)
_DEGRADING_SENSORS = ["sensor_3", "sensor_4", "sensor_7", "sensor_11", "sensor_12"]


@app.websocket("/ws/replay")
async def ws_replay(
    websocket: WebSocket,
    speed_ms: int = 400,
    total_steps: int = 150,
) -> None:
    """Streams a synthetic sensor degradation sequence for dashboard demos.

    Phases:
      Steps   0–40:  healthy operation  (scores stay low)
      Steps  40–90:  gradual degradation (scores trend up)
      Steps  90–150: clear anomaly zone  (scores consistently above threshold)

    Uses the real VAE scorer if loaded. Falls back to a simulated score
    that follows the same degradation pattern — so the demo works even
    before the VAE checkpoint exists.

    Query params:
      speed_ms    — milliseconds between readings (default 400)
      total_steps — how many readings to emit before looping (default 150)
    """
    await websocket.accept()
    logger.info("Replay WebSocket connected (speed=%dms, steps=%d)", speed_ms, total_steps)

    scorer = model_store.get_vae_scorer()
    n_sensors = len(_SENSOR_NAMES)
    window_size = scorer.window_size if scorer else 30
    buf: deque = deque(maxlen=window_size)
    threshold = float(scorer.threshold) if scorer else 0.05

    rng = np.random.default_rng(42)

    try:
        step = 0
        while True:
            # --- Generate synthetic sensor reading ---
            # Normalised values: healthy ≈ N(0,1), degradation adds drift
            t = step % total_steps
            degradation = max(0.0, (t - 40) / 50.0)   # ramps from 0 → 1 over steps 40–90
            degradation = min(degradation, 2.0)         # cap at 2× for stability

            reading = rng.normal(0.0, 1.0, n_sensors).tolist()
            for i, name in enumerate(_SENSOR_NAMES):
                if name in _DEGRADING_SENSORS:
                    reading[i] += degradation * rng.normal(1.5, 0.3)

            sensor_dict = {name: round(reading[i], 4) for i, name in enumerate(_SENSOR_NAMES)}
            buf.append(reading)

            # --- Score ---
            if scorer is not None and len(buf) == window_size:
                readings_np = np.array(list(buf), dtype=np.float32)
                loop = asyncio.get_event_loop()
                score = await loop.run_in_executor(
                    None, _score_ts_window, scorer, readings_np
                )
            else:
                # Simulated score: low baseline + degradation signal + noise
                base = float(rng.exponential(0.005))
                score = base + degradation * 0.04 + float(rng.normal(0, 0.003))
                score = max(0.0, score)

            await websocket.send_json({
                "step": t,
                "unit_id": 1,
                "anomaly_score": round(score, 6),
                "is_anomalous": bool(score > threshold),
                "threshold": round(threshold, 6),
                "sensors": sensor_dict,
                "phase": (
                    "healthy" if t < 40
                    else "degrading" if t < 90
                    else "anomalous"
                ),
            })

            await asyncio.sleep(speed_ms / 1000)
            step += 1

    except WebSocketDisconnect:
        logger.info("Replay WebSocket disconnected.")
    except Exception as e:
        logger.error("Replay WebSocket error: %s", e)
        await websocket.close(code=1011)
