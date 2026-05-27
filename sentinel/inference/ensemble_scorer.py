from __future__ import annotations

import numpy as np
import pandas as pd

from sentinel.inference.baseline import GaussianAnomalyScorer
from sentinel.inference.vae_scorer import VAEAnomalyScorer


class EnsembleScorer:
    """Late-fusion ensemble: VAE reconstruction error + Gaussian z-score.

    Each model produces a per-engine anomaly score on a different scale.
    We normalize both to [0,1] using bounds from healthy training data, then
    take a weighted average. A single threshold is applied to the fused score.

    Late fusion is used here (vs early fusion / joint training) because both
    models are already trained and use fundamentally different representations —
    combining at the score level avoids any retraining and keeps each model's
    internal calibration intact.

    Normalization uses min-max bounds from training healthy data. Test scores
    are clipped to [0,1] in case they exceed training bounds — this is expected
    for anomalous engines and is the correct behavior, not a bug.
    """

    def __init__(
        self,
        vae_scorer: VAEAnomalyScorer,
        gaussian_scorer: GaussianAnomalyScorer,
        threshold: float,
        vae_min: float,
        vae_max: float,
        gaussian_min: float,
        gaussian_max: float,
        vae_weight: float = 0.5,
    ) -> None:
        self.vae_scorer = vae_scorer
        self.gaussian_scorer = gaussian_scorer
        self.threshold = threshold
        self.vae_min = vae_min
        self.vae_max = vae_max
        self.gaussian_min = gaussian_min
        self.gaussian_max = gaussian_max
        self.vae_weight = vae_weight

    @classmethod
    def fit(
        cls,
        vae_scorer: VAEAnomalyScorer,
        gaussian_scorer: GaussianAnomalyScorer,
        healthy_df: pd.DataFrame,
        percentile: float = 99.0,
        vae_weight: float = 0.5,
    ) -> EnsembleScorer:
        """Compute normalization bounds from healthy data and calibrate threshold.

        healthy_df should be the raw (un-normalized) healthy training cycles —
        the same data used to fit both individual scorers.
        """
        vae_scores = vae_scorer.score_engines(healthy_df)
        gaussian_scores = cls._gaussian_score_engines(gaussian_scorer, healthy_df)

        vae_min = float(vae_scores.min())
        vae_max = float(vae_scores.max())
        gaussian_min = float(gaussian_scores.min())
        gaussian_max = float(gaussian_scores.max())

        scorer = cls(
            vae_scorer=vae_scorer,
            gaussian_scorer=gaussian_scorer,
            threshold=0.0,
            vae_min=vae_min,
            vae_max=vae_max,
            gaussian_min=gaussian_min,
            gaussian_max=gaussian_max,
            vae_weight=vae_weight,
        )

        fused_healthy = scorer._fuse(vae_scores, gaussian_scores)
        scorer.threshold = float(np.percentile(fused_healthy, percentile))
        return scorer

    def score_engines(self, df: pd.DataFrame) -> pd.Series:
        """Return fused anomaly score per engine unit, indexed by unit ID."""
        vae_scores = self.vae_scorer.score_engines(df)
        gaussian_scores = self._gaussian_score_engines(self.gaussian_scorer, df)
        common = vae_scores.index.intersection(gaussian_scores.index)
        return self._fuse(vae_scores[common], gaussian_scores[common])

    def predict_engines(self, df: pd.DataFrame) -> pd.Series:
        """Return boolean Series: True = engine predicted anomalous."""
        return self.score_engines(df) > self.threshold

    def _fuse(self, vae_scores: pd.Series, gaussian_scores: pd.Series) -> pd.Series:
        vae_norm = (vae_scores - self.vae_min) / (self.vae_max - self.vae_min + 1e-8)
        gaussian_norm = (gaussian_scores - self.gaussian_min) / (
            self.gaussian_max - self.gaussian_min + 1e-8
        )
        return (
            self.vae_weight * vae_norm.clip(0, 1)
            + (1 - self.vae_weight) * gaussian_norm.clip(0, 1)
        )

    @staticmethod
    def _gaussian_score_engines(
        scorer: GaussianAnomalyScorer, df: pd.DataFrame
    ) -> pd.Series:
        """Aggregate per-row Gaussian z-scores to per-engine max."""
        row_scores = scorer.score(df)
        return df.assign(_score=row_scores).groupby("unit")["_score"].max()
