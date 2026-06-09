"""Singleton model store: loads all SENTINEL models once at app startup.

Pattern: load-once-serve-many.
All models live in module-level variables initialized by `load_all()`.
FastAPI's lifespan hook calls `load_all()` before the first request arrives.
Subsequent requests read from these variables — no disk I/O per request.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import torch

logger = logging.getLogger(__name__)

# Module-level singletons — populated by load_all()
_vae_scorer = None
_audio_scorer = None


def load_all(
    vae_checkpoint: Path = Path("artifacts/vae/checkpoint.pt"),
    audio_scorer_path: Path = Path("artifacts/audio/scorer.pkl"),
    device: str = "cpu",
) -> dict[str, bool]:
    """Load all models into memory. Returns {model_name: loaded_ok} dict."""
    global _vae_scorer, _audio_scorer
    results: dict[str, bool] = {}

    # --- VAE time-series scorer ---
    if vae_checkpoint.exists():
        try:
            from sentinel.models.vae import VAE
            from sentinel.inference.vae_scorer import VAEAnomalyScorer

            ckpt = torch.load(vae_checkpoint, weights_only=False, map_location=device)
            model = VAE(**ckpt["model_config"])
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            _vae_scorer = VAEAnomalyScorer(
                model=model,
                threshold=ckpt["threshold"],
                mean=pd.Series(ckpt["mean"]),
                std=pd.Series(ckpt["std"]),
                sensor_cols=ckpt["sensor_cols"],
                window_size=ckpt["model_config"]["window_size"],
                use_delta_features=ckpt.get("use_delta_features", False),
            )
            logger.info("VAE scorer loaded from %s", vae_checkpoint)
            results["vae"] = True
        except Exception as e:
            logger.error("Failed to load VAE scorer: %s", e)
            results["vae"] = False
    else:
        logger.warning("VAE checkpoint not found at %s — skipping", vae_checkpoint)
        results["vae"] = False

    # --- Audio scorer ---
    if audio_scorer_path.exists():
        try:
            from sentinel.inference.audio_scorer import AudioAnomalyScorer

            _audio_scorer = AudioAnomalyScorer.load(audio_scorer_path, device=device)
            logger.info("Audio scorer loaded from %s", audio_scorer_path)
            results["audio"] = True
        except Exception as e:
            logger.error("Failed to load audio scorer: %s", e)
            results["audio"] = False
    else:
        logger.warning("Audio scorer not found at %s — skipping", audio_scorer_path)
        results["audio"] = False

    return results


def get_vae_scorer():
    """Return the loaded VAE scorer, or None if not available."""
    return _vae_scorer


def get_audio_scorer():
    """Return the loaded audio scorer, or None if not available."""
    return _audio_scorer
