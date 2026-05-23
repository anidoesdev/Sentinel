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






