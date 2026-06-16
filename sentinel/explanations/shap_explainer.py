"""SHAP explanations for the Gaussian anomaly scorer.

KernelSHAP treats the scorer as a black box f(X) -> score and estimates
Shapley values by sampling coalitions of features. The result is a
[n_samples, n_sensors] matrix where each value answers:

    "How much did this sensor contribute to this anomaly score,
     relative to the expected score on healthy data?"

Positive = sensor pushed the score up (anomalous direction).
Negative = sensor actually pulled the score down (normal for this reading).
Values sum to: actual_score - E[score over healthy background].
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import shap

from sentinel.inference.baseline import GaussianAnomalyScorer

logger = logging.getLogger(__name__)


class GaussianSHAPExplainer:
    """KernelSHAP wrapper for GaussianAnomalyScorer.

    Background data defines the baseline expectation E[f(X)]. KernelSHAP
    summarises this with k-means to 100 representative points — running the
    full training set as background would be computationally prohibitive
    (KernelSHAP calls the model O(n_background * n_coalitions) times).
    """

    def __init__(
        self,
        scorer: GaussianAnomalyScorer,
        background_df: pd.DataFrame,
        n_background: int = 100,
    ) -> None:
        self.scorer = scorer
        self.sensor_cols = scorer.sensor_cols

        background_np = background_df[self.sensor_cols].values.astype(np.float64)
        k = min(n_background, len(background_np))
        logger.info("Summarising background with k-means (k=%d)...", k)

        # shap.kmeans returns a DenseData summary object — the KernelExplainer
        # treats each centroid as a "missing feature" stand-in when building coalitions
        self.explainer = shap.KernelExplainer(
            model=self._score_fn,
            data=shap.kmeans(background_np, k),
        )
        logger.info("KernelSHAP explainer ready.")

    def _score_fn(self, X: np.ndarray) -> np.ndarray:
        """Bridge: numpy [n, n_sensors] → scorer → anomaly scores [n]."""
        df = pd.DataFrame(X, columns=self.sensor_cols)
        return self.scorer.score(df).values.astype(np.float64)

    def explain(
        self,
        windows: pd.DataFrame,
        n_samples: int = 100,
        silent: bool = True,
    ) -> np.ndarray:
        """Return SHAP values for every row in windows.

        Args:
            windows:  DataFrame with sensor columns (typically anomalous rows).
            n_samples: coalition samples per explanation. Higher = more accurate,
                       slower. 100 is a good default; use 500 for final results.

        Returns:
            shap_values: [n_rows, n_sensors] float array.
        """
        X = windows[self.sensor_cols].values.astype(np.float64)
        values = self.explainer.shap_values(X, nsamples=n_samples, silent=silent)
        logger.info("SHAP values computed for %d rows.", len(windows))
        return np.array(values)

    def feature_importance(self, shap_values: np.ndarray) -> pd.DataFrame:
        """Mean absolute SHAP value per sensor, sorted descending.

        This is the standard SHAP global importance summary — it tells you
        which sensors matter most on average across the explained windows.
        """
        mean_abs = np.abs(shap_values).mean(axis=0)
        return (
            pd.DataFrame({"sensor": self.sensor_cols, "mean_abs_shap": mean_abs})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )
