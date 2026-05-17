# tests/data/test_cmapss.py
import pandas as pd
import numpy as np
import pytest
from sentinel.data.cmapss import drop_low_variance, get_healthy_cycle, COLUMNS

@pytest.fixture
def sample_df() -> pd.DataFrame:
    # build a small synthetic dataframe that looks like CMAPSS
    # 3 units, 20 cycles each, all sensor values random except sensor_1 = constant
    rng = np.random.default_rng(42)
    units = [1,2,3]
    cycles_per_unit = 20
    
    rows = []
    for unit in units:
        for cycle in range(1,cycles_per_unit+1):
            row = [unit,cycle]
            row += [1.0,1.0,1.0]
            row += [0.0]
            row += rng.normal(0,1,20).tolist()
            rows.append(row)
    return pd.DataFrame(rows, columns=COLUMNS)

def test_drop_low_variance_removes_constant_column(sample_df):
    # sensor_1 is constant — should be in dropped list
    # remaining df should not contain sensor_1
    cleaned_df, dropped = drop_low_variance(sample_df)
    assert "sensor_1" in dropped
    assert "sensor_1" not in cleaned_df.columns
    

def test_drop_low_variance_returns_correct_types(sample_df):
    # return value is (DataFrame, list)
    cleaned_df, dropped = drop_low_variance(sample_df)
    assert isinstance(cleaned_df, pd.DataFrame)
    assert isinstance(dropped, list)

def test_get_healthy_cycle_removes_tail(sample_df):
    # with tail_cycles=5 and 20 cycles per unit,
    # max cycle in result should be 15, not 20
    res = get_healthy_cycle(sample_df,5)
    assert res["cycle"].max() == 15

def test_get_healthy_cycle_preserves_all_units(sample_df):
    # all 3 units should still be present after filtering
    res = get_healthy_cycle(sample_df,5)
    assert set(res["unit"].unique()) == {1,2,3}
