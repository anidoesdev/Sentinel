# tests/data/test_cmapss.py
import pandas as pd
import numpy as np
import pytest
from sentinel.data.cmapss import drop_low_variance, get_healthy_cycle, COLUMNS

@pytest.fixture
def sample_df() -> pd.DataFrame:
    # build a small synthetic dataframe that looks like CMAPSS
    # 3 units, 20 cycles each, all sensor values random except sensor_1 = constant
    ...

def test_drop_low_variance_removes_constant_column(sample_df):
    # sensor_1 is constant — should be in dropped list
    # remaining df should not contain sensor_1
    ...

def test_drop_low_variance_returns_correct_types(sample_df):
    # return value is (DataFrame, list)
    ...

def test_get_healthy_cycle_removes_tail(sample_df):
    # with tail_cycles=5 and 20 cycles per unit,
    # max cycle in result should be 15, not 20
    ...

def test_get_healthy_cycle_preserves_all_units(sample_df):
    # all 3 units should still be present after filtering
    ...
