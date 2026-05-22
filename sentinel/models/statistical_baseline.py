import pandas as pd
import numpy as np

class GaussianBaseline:
    def fit(self, df: pd.DataFrame, sensor_cols: list[str]) -> None:
        # compute and store mean and std per sensor
        self.mean = {col: df[col].mean() for col in sensor_cols}
        self.std = {col: df[col].std() for col in sensor_cols}
        self.sensor_cols_ = sensor_cols

    def score(self, df: pd.DataFrame, sensor_cols: list[str], window: int = 5) -> pd.Series:
        # rolling mean smooth, then max z-score across sensors per row
        smoothed = {col: df.groupby("unit")[col].transform(lambda x: x.rolling(window, min_periods=1).mean()) for col in sensor_cols}
        z_score = pd.DataFrame({
            col: (smoothed[col] - self.mean[col]).abs() / self.std[col]
            for col in sensor_cols
        })
        return z_score.max(axis=1)

    def threshold(self, scores: pd.Series, percentile: float = 99.0) -> float:
        # return the percentile value of scores
        return np.percentile(scores,percentile)
    
