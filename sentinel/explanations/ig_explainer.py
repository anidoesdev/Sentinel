"""Integrated Gradients explanations for the VAE anomaly scorer.

IG produces a [window_size, n_sensors] attribution map answering:
    "Compared to a healthy baseline window, which sensor at which timestep
     drove the high reconstruction error?"

The math (Riemann sum approximation of the path integral):
    IG_i(x) = (x_i - b_i) * (1/m) * Σ_{k=1}^{m} ∂F(b + (k/m)(x-b)) / ∂x_i

Where:
    x  = anomalous input window
    b  = baseline (mean healthy window)
    F  = reconstruction error (MSE between input and VAE output)
    m  = n_steps (50 is accurate; 200 for publication quality)

Completeness guarantee: sum(IG) ≈ F(x) - F(baseline).
This means the attributions exactly account for the full difference in
reconstruction error between the anomalous window and a healthy one.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch

from sentinel.models.vae import VAE

logger = logging.getLogger(__name__)


class VAEIntegratedGradients:
    """Integrated Gradients for VAE reconstruction error.

    Uses deterministic encoding (mu only, no reparameterize sampling) so
    the score function is deterministic — required for IG to be meaningful.
    Stochastic forward passes would give different gradients each time,
    violating the completeness axiom.
    """

    def __init__(
        self,
        model: VAE,
        baseline: torch.Tensor,
        n_steps: int = 50,
    ) -> None:
        """
        Args:
            model:    trained VAE, expected in eval mode.
            baseline: [1, window_size, n_sensors] — healthy reference window.
            n_steps:  Riemann sum steps. 50 gives <2% completeness error.
        """
        self.model = model
        self.model.eval()
        self.baseline = baseline.float()
        self.n_steps = n_steps

    @classmethod
    def from_healthy_windows(
        cls,
        model: VAE,
        windows: np.ndarray,
        n_steps: int = 50,
    ) -> "VAEIntegratedGradients":
        """Construct explainer with mean healthy window as baseline.

        The mean healthy window is the natural IG baseline for sensor data —
        it represents "what a normal engine looks like." IG attributions then
        answer "relative to a healthy engine, what made this window anomalous?"

        windows: [N, window_size, n_sensors] normalised numpy array.
        """
        baseline_np = windows.mean(axis=0, keepdims=True)  # [1, W, S]
        baseline = torch.from_numpy(baseline_np.astype(np.float32))
        logger.info("IG baseline: mean of %d healthy windows.", len(windows))
        return cls(model=model, baseline=baseline, n_steps=n_steps)

    def explain(self, x: torch.Tensor) -> np.ndarray:
        """Compute IG attribution map for a single window.

        Args:
            x: [1, window_size, n_sensors] normalised sensor window.

        Returns:
            ig: [window_size, n_sensors] attribution map.
                Positive: sensor-timestep pushed reconstruction error up.
                Negative: sensor-timestep was easier to reconstruct than baseline.
        """
        self.model.eval()
        x = x.float()

        # Riemann sum over n_steps interpolated inputs between baseline and x.
        # Exclude α=0 (would divide by zero in completeness) and include α=1.
        alphas = torch.linspace(0.0, 1.0, self.n_steps + 1)[1:]  # [n_steps]

        accumulated_grads = torch.zeros_like(x)

        for alpha in alphas:
            # Interpolate: move α-fraction of the way from baseline to x
            x_interp = (self.baseline + alpha * (x - self.baseline)).detach()
            x_interp.requires_grad_(True)

            # Deterministic forward: mu → decoder (skip reparameterize)
            mu, _ = self.model.encoder(x_interp)
            x_hat = self.model.decoder(mu)

            # Scalar reconstruction error
            recon_error = ((x_interp - x_hat) ** 2).mean()
            recon_error.backward()

            accumulated_grads = accumulated_grads + x_interp.grad.detach()

        # Average gradients over all steps, then scale by (x - baseline).
        # This is the discrete approximation of the path integral.
        avg_grads = accumulated_grads / self.n_steps          # [1, W, S]
        ig = ((x - self.baseline) * avg_grads).squeeze(0)    # [W, S]
        ig_np = ig.numpy()

        # Completeness check: sum(ig) should ≈ f(x) - f(baseline)
        self._log_completeness(x, ig_np)

        return ig_np  # [window_size, n_sensors]

    def _log_completeness(self, x: torch.Tensor, ig_np: np.ndarray) -> None:
        """Log the completeness gap as a sanity check. Should be < 5%."""
        with torch.no_grad():
            mu_x, _ = self.model.encoder(x)
            score_x = float(((x - self.model.decoder(mu_x)) ** 2).mean())

            mu_b, _ = self.model.encoder(self.baseline)
            score_b = float(((self.baseline - self.model.decoder(mu_b)) ** 2).mean())

        ig_sum = float(ig_np.sum())
        expected_diff = score_x - score_b
        gap_pct = abs(ig_sum - expected_diff) / (abs(expected_diff) + 1e-8) * 100
        logger.debug(
            "IG completeness: sum(ig)=%.5f  f(x)-f(b)=%.5f  gap=%.1f%%",
            ig_sum, expected_diff, gap_pct,
        )

    def sensor_importance(
        self, ig: np.ndarray, sensor_cols: list[str]
    ) -> pd.DataFrame:
        """Sum absolute IG over timesteps → per-sensor total attribution.

        Collapsing the time dimension gives a single importance value per
        sensor — useful for comparing sensors across many anomalous windows.
        """
        total = np.abs(ig).sum(axis=0)  # [n_sensors]
        return (
            pd.DataFrame({"sensor": sensor_cols, "total_ig": total})
            .sort_values("total_ig", ascending=False)
            .reset_index(drop=True)
        )
