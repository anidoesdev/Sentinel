from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from sentinel.models.vae import VAE


class VAEAnomalyScorer:
    """Reconstruction-error anomaly scorer backed by a trained VAE.

    Scoring intuition: the VAE is trained only on healthy windows, so it learns
    to compress and reconstruct normal sensor patterns. When it sees a degraded
    window it hasn't learned, the decoder produces a poor reconstruction →
    high MSE per window → high anomaly score for that engine.

    Threshold is calibrated at the 99th percentile of healthy-window errors so
    that ~1% of normal windows are false positives — a conservative choice for
    safety-critical systems where missing a fault is more costly than a false alarm.
    """

    def __init__(
        self,
        model: VAE,
        threshold: float,
        mean: pd.Series,
        std: pd.Series,
        sensor_cols: list[str],
        window_size: int = 30,
    ) -> None:
        self.model = model
        self.threshold = threshold
        self.mean = mean
        self.std = std
        self.sensor_cols = sensor_cols
        self.window_size = window_size

    @classmethod
    def fit(
        cls,
        model: VAE,
        healthy_df: pd.DataFrame,
        mean: pd.Series,
        std: pd.Series,
        sensor_cols: list[str],
        window_size: int = 30,
        percentile: float = 99.0,
    ) -> VAEAnomalyScorer:
        """Calibrate the anomaly threshold from healthy reconstruction errors.

        healthy_df should be raw (un-normalized); normalization is applied internally
        using `mean` and `std`, which are stored for use at inference time.
        """
        scorer = cls(
            model=model, threshold=0.0, mean=mean, std=std,
            sensor_cols=sensor_cols, window_size=window_size,
        )
        normed = cls._normalize_df(healthy_df, mean, std, sensor_cols)
        windows = cls._make_windows(normed, sensor_cols, window_size)
        errors = scorer._score_windows(windows)
        scorer.threshold = float(np.percentile(errors, percentile))
        return scorer

    def score_engines(self, df: pd.DataFrame) -> pd.Series:
        """Return a Series of max-window reconstruction error per engine unit.

        Max is used instead of mean so that a single severely degraded cycle
        is enough to trigger a high score — conservative for fault detection.
        """
        scores: dict[int, float] = {}
        for unit_id, group in df.groupby("unit"):
            normed = self._normalize_df(group, self.mean, self.std, self.sensor_cols)
            if len(normed) < self.window_size:
                continue
            windows = self._make_windows(normed, self.sensor_cols, self.window_size)
            window_errors = self._score_windows(windows)
            scores[int(unit_id)] = float(window_errors.max())
        return pd.Series(scores)

    def predict_engines(self, df: pd.DataFrame) -> pd.Series:
        """Return boolean Series: True = engine predicted anomalous."""
        return self.score_engines(df) > self.threshold

    def _score_windows(self, windows: np.ndarray) -> np.ndarray:
        """Compute per-window MSE reconstruction error.

        windows: [N, window_size, input_dim]
        returns: [N] array of scalar MSE values
        """
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(windows, dtype=torch.float32)
            recon, _, _ = self.model(x)
            # Mean over both time and feature dims → one scalar per window.
            errors = (x - recon).pow(2).mean(dim=(1, 2))
        return errors.numpy()

    @staticmethod
    def _normalize_df(
        df: pd.DataFrame, mean: pd.Series, std: pd.Series, sensor_cols: list[str]
    ) -> pd.DataFrame:
        out = df.copy()
        # +1e-8 guards against zero-std sensors that slipped through variance filtering.
        out[sensor_cols] = (out[sensor_cols] - mean) / (std + 1e-8)
        return out

    @staticmethod
    def _make_windows(
        df: pd.DataFrame, sensor_cols: list[str], window_size: int
    ) -> np.ndarray:
        data = df[sensor_cols].values
        if len(data) < window_size:
            return np.empty((0, window_size, len(sensor_cols)))
        return np.stack([
            data[i: i + window_size]
            for i in range(len(data) - window_size + 1)
        ])
