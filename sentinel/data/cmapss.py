from pathlib import Path
import pandas as pd
import numpy as np
from pydantic import BaseModel

COLUMNS = (
    ["unit","cycle"]
    + [f"setting_{i}" for i in range(1,4)]
    + [f"sensor_{i}" for i in range(1,22)]
)

class CMAPSSConfig(BaseModel):
    data_dir: Path
    fd_id: int = 1
    drop_cols: list[str] = []

def load_raw(config: CMAPSSConfig, split: str = "train") -> pd.DataFrame:
    # load train_FD00{fd_id}.txt, assign COLUMNS, return dataframe
    base_path = config.data_dir/"raw"
    path = base_path/f"{split}_FD00{config.fd_id}.txt"
    data = pd.read_csv(path,sep=r"\s+",header=None,names=COLUMNS)
    return data

def drop_low_variance(df: pd.DataFrame, threshold: float = 1e-4) -> tuple[pd.DataFrame, list[str]]:
    # drop sensor columns where variance < threshold 
    # return cleaned df and list of dropped column names
    sensor_cols = [f"sensor_{i}"for i in range(1,22)]
    dropped_cols = []
    for col in sensor_cols:
        if df[col].var() < threshold:
            dropped_cols.append(col)
    return (df.drop(columns=dropped_cols),dropped_cols)

    
def get_healthy_cycle(df: pd.DataFrame, tail_cycles: int = 30) -> pd.DataFrame:
    # for each unit, drop the last `tail_cycles` cycles
    # this is your "normal" training data 
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    return df[df["cycle"] <= max_cycle - tail_cycles]

def make_windows(df: pd.DataFrame, sensor_cols: list[str], window_size: int = 30) -> np.ndarray:
    # for each unit, slide a window of size `window_size` with stride 1
    # return array of shape [num_windows, window_size, len(sensor_cols)]
    windows = []
    for _, group in df.groupby("unit"):
        data = group[sensor_cols].values
        for i in range(len(data) - window_size + 1):
            windows.append(data[i : i + window_size])
    return np.stack(windows)


def add_delta_features(
    df: pd.DataFrame, sensor_cols: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    """Add per-cycle first-difference features, computed per engine unit.

    Gradual degradation is nearly invisible in absolute sensor values — a drift
    of 0.3 sigma per window looks healthy. But in the delta space, that same
    drift is a sustained non-zero signal the VAE has never seen in training.

    The first cycle of each unit gets delta=0 (no prior reading).
    Returns (df_with_deltas, delta_col_names).
    """
    out = df.copy()
    delta_cols = [f"d_{col}" for col in sensor_cols]
    for col, dcol in zip(sensor_cols, delta_cols):
        out[dcol] = out.groupby("unit")[col].diff().fillna(0.0)
    return out, delta_cols


def normalize(df: pd.DataFrame, sensor_cols: list[str],
              mean: pd.Series | None = None, 
              std: pd.Series | None = None) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    # if mean/std not provided, compute from df (training mode)
    # if provided, apply them (inference mode)
    # return normalized df, mean, std
    if mean is None:
        mean = df[sensor_cols].mean()
    if std is None:
        std = df[sensor_cols].std()
        
    df = df.copy()
    df[sensor_cols] = (df[sensor_cols] - mean) / std
    
    return df,mean,std
    


# config = CMAPSSConfig(data_dir=Path("data"), fd_id=1)
# df, dropped = drop_low_variance(load_raw(config))
# sensor_cols = [c for c in df.columns if c.startswith("sensor_") and c not in dropped]
# healthy = get_healthy_cycle(df)
# healthy, mean, std = normalize(healthy,sensor_cols)
# windows = make_windows(healthy, sensor_cols)
# print(windows.shape)



