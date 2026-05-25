from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from sentinel.models.vae import VAE


class VAEAnomalyScorer:
    """Reconstruction-error anomaly scorer backed by a trained VAE.

    Scoring intuition: the VAE learns to compress and reconstruct healthy sensor
    windows. Degraded windows land outside the learned distribution, so the
    decoder reconstructs them poorly → high MSE = anomaly signal.

    With use_delta_features=True the model also sees the cycle-to-cycle rate of
    change of each sensor. Gradual degradation that's invisible in raw values
    (a 0.3-sigma drift per window) shows up as a sustained non-zero delta that
    the VAE has never seen in training and reconstructs poorly.
    """

    def __init__(
        self,
        model: VAE,
        threshold: float,
        mean: pd.Series,
        std: pd.Series,
        sensor_cols: list[str],
        window_size: int = 30,
        use_delta_features: bool = False,
    ) -> None:
        self.model = model
        self.threshold = threshold
        self.mean = mean
        self.std = std
        self.sensor_cols = sensor_cols
        self.window_size = window_size
        self.use_delta_features = use_delta_features
        # feature_cols is what the model actually ingests: raw + optional deltas.
        if use_delta_features:
            self.feature_cols: list[str] = sensor_cols + [f"d_{c}" for c in sensor_cols]
        else:
            self.feature_cols = list(sensor_cols)

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
        use_delta_features: bool = False,
    ) -> VAEAnomalyScorer:
        """Calibrate the anomaly threshold from healthy reconstruction errors.

        healthy_df must be raw (un-normalized). Normalization and delta computation
        are applied internally using the stored mean/std and sensor_cols.
        """
        scorer = cls(
            model=model, threshold=0.0, mean=mean, std=std,
            sensor_cols=sensor_cols, window_size=window_size,
            use_delta_features=use_delta_features,
        )

        all_windows: list[np.ndarray] = []
        for _, group in healthy_df.groupby("unit"):
            group_feat = scorer._add_deltas(group)
            normed = cls._normalize_df(group_feat, mean, std, scorer.feature_cols)
            if len(normed) >= window_size:
                windows = cls._make_windows(normed, scorer.feature_cols, window_size)
                all_windows.append(windows)

        if not all_windows:
            raise ValueError("No healthy windows found — check window_size vs data length.")

        errors = scorer._score_windows(np.concatenate(all_windows, axis=0))
        scorer.threshold = float(np.percentile(errors, percentile))
        return scorer

    def score_engines(self, df: pd.DataFrame) -> pd.Series:
        """Return max reconstruction error per engine unit over all its windows.

        Max is used so that even a single severely degraded window can flag an
        engine — conservative choice for fault detection.
        """
        scores: dict[int, float] = {}
        for unit_id, group in df.groupby("unit"):
            group_feat = self._add_deltas(group)
            normed = self._normalize_df(group_feat, self.mean, self.std, self.feature_cols)
            if len(normed) < self.window_size:
                continue
            windows = self._make_windows(normed, self.feature_cols, self.window_size)
            window_errors = self._score_windows(windows)
            scores[int(unit_id)] = float(window_errors.max())
        return pd.Series(scores)

    def predict_engines(self, df: pd.DataFrame) -> pd.Series:
        """Return boolean Series: True = engine predicted anomalous."""
        return self.score_engines(df) > self.threshold

    def _add_deltas(self, group: pd.DataFrame) -> pd.DataFrame:
        """Add first-difference features for a single-unit DataFrame."""
        if not self.use_delta_features:
            return group
        out = group.copy()
        for col in self.sensor_cols:
            # diff() on a single-unit group; first row gets NaN → fill with 0.
            out[f"d_{col}"] = out[col].diff().fillna(0.0)
        return out

    def _score_windows(self, windows: np.ndarray) -> np.ndarray:
        """Compute per-window MSE reconstruction error.

        windows: [N, window_size, input_dim]
        returns: [N] array of scalar MSE values
        """
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(windows, dtype=torch.float32)
            recon, _, _ = self.model(x)
            errors = (x - recon).pow(2).mean(dim=(1, 2))
        return errors.numpy()

    @staticmethod
    def _normalize_df(
        df: pd.DataFrame, mean: pd.Series, std: pd.Series, feature_cols: list[str]
    ) -> pd.DataFrame:
        out = df.copy()
        out[feature_cols] = (out[feature_cols] - mean) / (std + 1e-8)
        return out

    @staticmethod
    def _make_windows(
        df: pd.DataFrame, feature_cols: list[str], window_size: int
    ) -> np.ndarray:
        data = df[feature_cols].values
        if len(data) < window_size:
            return np.empty((0, window_size, len(feature_cols)))
        return np.stack([
            data[i: i + window_size]
            for i in range(len(data) - window_size + 1)
        ])
