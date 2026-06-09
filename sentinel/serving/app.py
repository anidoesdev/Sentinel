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

    Inference is CPU-bound, so it runs in a thread pool to avoid blocking the
    event loop and starving other concurrent requests.
    """
    scorer = model_store.get_vae_scorer()
    if scorer is None:
        raise HTTPException(status_code=503, detail="VAE scorer not loaded")

    readings_np = np.array(window.readings, dtype=np.float32)  # [timesteps, sensors]

    # run_in_executor: move the blocking torch inference off the event loop thread
    loop = asyncio.get_event_loop()
    score = await loop.run_in_executor(None, _score_ts_window, scorer, readings_np)

    return AnomalyScore(
        unit_id=window.unit_id,
        anomaly_score=float(score),
        is_anomalous=bool(score > scorer.threshold),
        threshold=float(scorer.threshold),
        modality="timeseries",
    )


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

    return AnomalyScore(
        unit_id=None,
        anomaly_score=float(score),
        is_anomalous=bool(score > scorer.threshold),
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
