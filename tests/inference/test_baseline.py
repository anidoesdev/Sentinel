import pandas as pd
import numpy as np
import pytest
from sentinel.inference.baseline import GaussianAnomalyScorer
from sentinel.data.cmapss import COLUMNS

@pytest.fixture
def health_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    units = [1, 2, 3]
    cycles_per_unit = 20
    
    rows = []
    for unit in units:
        for cycle in range(1, cycles_per_unit + 1):
            row = [unit, cycle, 1.0, 1.0, 1.0]
            row += rng.normal(0, 1, 21).tolist()
            rows.append(row)
    return pd.DataFrame(rows, columns=COLUMNS)

def test_scorer_fit_creates_instance(health_data):
    assert GaussianAnomalyScorer.fit(health_data)

def test_scorer_healthy_data_low_scores(health_data):
    fit_data = GaussianAnomalyScorer.fit(healthy_df=health_data)
    scores = fit_data.score(health_data)
    assert np.percentile(scores,95) < 3.0

def test_scorer_anomalous_data_high_scores(health_data):
    #create sythetic anomalous data 
    fit_data = GaussianAnomalyScorer.fit(healthy_df=health_data)
    anomalous_df = health_data.copy()
    sensor_col = "sensor_2"
    anomalous_df[sensor_col] += 5 * fit_data.sigma[sensor_col]
    scores = fit_data.score(anomalous_df)
    assert (scores > fit_data.threshold).any()
