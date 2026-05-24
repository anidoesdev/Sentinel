import numpy as np
import pandas as pd
import pytest
import torch

from sentinel.inference.vae_scorer import VAEAnomalyScorer
from sentinel.models.vae import VAE

# Small constants so tests run in milliseconds.
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 6)]  # 5 sensors
INPUT_DIM = len(SENSOR_COLS)
WINDOW_SIZE = 5


def _make_df(
    n_units: int = 3,
    n_cycles: int = 20,
    shift: float = 0.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a synthetic sensor DataFrame with optional mean shift."""
    rng = np.random.default_rng(seed)
    rows = []
    for unit in range(1, n_units + 1):
        for cycle in range(1, n_cycles + 1):
            row: dict = {"unit": unit, "cycle": cycle}
            for col in SENSOR_COLS:
                row[col] = float(rng.normal(0, 1)) + shift
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def tiny_model() -> VAE:
    torch.manual_seed(0)
    return VAE(input_dim=INPUT_DIM, latent_dim=4, window_size=WINDOW_SIZE)


@pytest.fixture
def healthy_df() -> pd.DataFrame:
    return _make_df()


def _fit(model: VAE, df: pd.DataFrame) -> VAEAnomalyScorer:
    mean = df[SENSOR_COLS].mean()
    std = df[SENSOR_COLS].std()
    return VAEAnomalyScorer.fit(
        model=model,
        healthy_df=df,
        mean=mean,
        std=std,
        sensor_cols=SENSOR_COLS,
        window_size=WINDOW_SIZE,
    )


def test_fit_creates_instance(tiny_model, healthy_df):
    scorer = _fit(tiny_model, healthy_df)
    assert isinstance(scorer, VAEAnomalyScorer)


def test_fit_sets_positive_threshold(tiny_model, healthy_df):
    scorer = _fit(tiny_model, healthy_df)
    assert scorer.threshold > 0.0


def test_score_engines_returns_series_indexed_by_unit(tiny_model, healthy_df):
    scorer = _fit(tiny_model, healthy_df)
    scores = scorer.score_engines(healthy_df)
    assert isinstance(scores, pd.Series)
    assert set(scores.index) == {1, 2, 3}


def test_score_engines_all_non_negative(tiny_model, healthy_df):
    scorer = _fit(tiny_model, healthy_df)
    scores = scorer.score_engines(healthy_df)
    assert (scores >= 0).all()


def test_anomalous_data_scores_higher_than_healthy(tiny_model, healthy_df):
    scorer = _fit(tiny_model, healthy_df)
    # A 10-sigma shift is so far out-of-distribution the model reconstructs
    # it poorly regardless of random weight initialization.
    anomalous_df = _make_df(shift=10.0, seed=99)
    healthy_scores = scorer.score_engines(healthy_df)
    anomalous_scores = scorer.score_engines(anomalous_df)
    assert anomalous_scores.mean() > healthy_scores.mean()


def test_predict_engines_returns_bool_series(tiny_model, healthy_df):
    scorer = _fit(tiny_model, healthy_df)
    preds = scorer.predict_engines(healthy_df)
    assert isinstance(preds, pd.Series)
    assert preds.dtype == bool


def test_unit_with_too_few_cycles_is_skipped(tiny_model, healthy_df):
    # A unit with fewer cycles than window_size should be silently skipped,
    # not raise an error.
    short_df = _make_df(n_cycles=WINDOW_SIZE - 1, n_units=1)
    scorer = _fit(tiny_model, healthy_df)
    scores = scorer.score_engines(short_df)
    assert len(scores) == 0
