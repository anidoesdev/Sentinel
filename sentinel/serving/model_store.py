"""Singleton model store: loads all SENTINEL models once at app startup.

Pattern: load-once-serve-many.
All models live in module-level variables initialized by `load_all()`.
FastAPI's lifespan hook calls `load_all()` before the first request arrives.

Shadow deployment:
  A "challenger" VAE checkpoint can be loaded alongside the "champion".
  Both score every request, but only the champion's result is returned to
  the client. The challenger's score is written to TimescaleDB (is_shadow=True)
  for offline comparison. This is the standard way to evaluate a new model
  version on real traffic without risk.

  To activate: place a checkpoint at artifacts/vae/challenger.pt
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import torch

logger = logging.getLogger(__name__)

_vae_scorer = None        # champion
_vae_challenger = None    # shadow — optional
_audio_scorer = None


def _load_vae(path: Path, device: str):
    from sentinel.models.vae import VAE
    from sentinel.inference.vae_scorer import VAEAnomalyScorer

    ckpt = torch.load(path, weights_only=False, map_location=device)
    model = VAE(**ckpt["model_config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return VAEAnomalyScorer(
        model=model,
        threshold=ckpt["threshold"],
        mean=pd.Series(ckpt["mean"]),
        std=pd.Series(ckpt["std"]),
        sensor_cols=ckpt["sensor_cols"],
        window_size=ckpt["model_config"]["window_size"],
        use_delta_features=ckpt.get("use_delta_features", False),
    )


def load_all(
    vae_checkpoint: Path = Path("artifacts/vae/checkpoint.pt"),
    vae_challenger_checkpoint: Path = Path("artifacts/vae/challenger.pt"),
    audio_scorer_path: Path = Path("artifacts/audio/scorer.pkl"),
    device: str = "cpu",
) -> dict[str, bool]:
    """Load all models. Returns {model_name: loaded_ok}."""
    global _vae_scorer, _vae_challenger, _audio_scorer
    results: dict[str, bool] = {}

    # Champion VAE
    if vae_checkpoint.exists():
        try:
            _vae_scorer = _load_vae(vae_checkpoint, device)
            logger.info("VAE champion loaded from %s", vae_checkpoint)
            results["vae"] = True
        except Exception as e:
            logger.error("Failed to load VAE champion: %s", e)
            results["vae"] = False
    else:
        logger.warning("VAE champion checkpoint not found: %s", vae_checkpoint)
        results["vae"] = False

    # Challenger VAE (shadow deployment — optional)
    if vae_challenger_checkpoint.exists():
        try:
            _vae_challenger = _load_vae(vae_challenger_checkpoint, device)
            logger.info("VAE challenger loaded (shadow mode) from %s", vae_challenger_checkpoint)
            results["vae_challenger"] = True
        except Exception as e:
            logger.warning("Failed to load VAE challenger (shadow disabled): %s", e)
            results["vae_challenger"] = False
    else:
        results["vae_challenger"] = False  # silent — challenger is optional

    # Audio scorer
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
        logger.warning("Audio scorer not found: %s", audio_scorer_path)
        results["audio"] = False

    return results


def get_vae_scorer():
    return _vae_scorer

def get_vae_challenger():
    """Returns the shadow challenger scorer, or None if not loaded."""
    return _vae_challenger

def get_audio_scorer():
    return _audio_scorer
