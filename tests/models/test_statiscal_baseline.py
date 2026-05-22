import pandas as pd
import numpy as np 
from sentinel.models.statistical_baseline import GaussianBaseline
import pytest
from sentinel.data.cmapss import COLUMNS

@pytest.fixture
def health_data():
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

def test_fit(health_data):
    baseline = GaussianBaseline()
    sensor_cols = [f"sensor_{i}" for i in range(1,22)]
    baseline.fit(health_data,sensor_cols=sensor_cols)
    assert isinstance(baseline.mean,dict)
    assert set(baseline.mean.keys()) == set(sensor_cols)

def test_score(health_data):
    baseline = GaussianBaseline()
    sensor_cols = [f"sensor_{i}" for i in range(1,22)]
    baseline.fit(health_data,sensor_cols)
    result = baseline.score(health_data,sensor_cols=sensor_cols)
    assert isinstance(result,pd.Series) 

def test_threshold(health_data):
    baseline = GaussianBaseline()
    sensor_cols = [f"sensor_{i}" for i in range(1,22)]
    baseline.fit(health_data,sensor_cols=sensor_cols)
    scores = baseline.score(health_data,sensor_cols=sensor_cols)
    result = baseline.threshold(scores)
    assert scores.min() <= result <= scores.max()