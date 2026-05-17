from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class GaussianAnomalyScorer:
    """Fit on healthy data, score new readings."""
    
    sensor_cols: list[str]
    mu: dict[str, float]  # mean per sensor
    sigma: dict[str, float]  # std per sensor
    threshold: float = 3.0
    
    @classmethod
    def fit(cls, healthy_df: pd.DataFrame, threshold: float = 3.0) -> "GaussianAnomalyScorer":
        # compute mu, sigma for each sensor_col
        # return instance
        sensor_cols = [col for col in healthy_df.columns if col.startswith("sensor_")]
        mu = {col: healthy_df[col].mean() for col in sensor_cols}
        sigma = {col: healthy_df[col].std() for col in sensor_cols}
        return cls(sensor_cols=sensor_cols,mu=mu,sigma=sigma,threshold=threshold)
    
    def score(self, df: pd.DataFrame) -> pd.Series:
        # compute z-scores for each sensor, then max(|z|) per row
        # return Series of anomaly scores
        z_scores = [np.abs((df[col]-self.mu[col])/self.sigma[col]) for col in self.sensor_cols]
        z_df = pd.concat(z_scores,axis=1)
        return z_df.max(axis=1)
    
    def predict(self, df: pd.DataFrame) -> pd.Series:
        # return boolean: True if anomalous (score > threshold), False otherwise
        scores = self.score(df)
        return scores > self.threshold