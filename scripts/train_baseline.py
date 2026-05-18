# scripts/train_baseline.py
"""Train and evaluate statistical baseline on CMAPSS FD001."""
import logging
from pathlib import Path
from sentinel.data.cmapss import load_raw, drop_low_variance, get_healthy_cycle, CMAPSSConfig
from sentinel.inference.baseline import GaussianAnomalyScorer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    data_dir = Path("data")
    config = CMAPSSConfig(data_dir=data_dir, fd_id=1)
    
    logger.info("Loading raw CMAPSS data...")
    train_df = load_raw(config)
    logger.info(f"Loaded {len(train_df)} rows")
    
    logger.info("Dropping low-variance sensors...")
    train_df, dropped = drop_low_variance(train_df)
    logger.info(f"Dropped sensors: {dropped}")
    
    logger.info("Extracting healthy cycles...")
    healthy_df = get_healthy_cycle(train_df, tail_cycles=30)
    logger.info(f"Healthy cycles: {len(healthy_df)} rows")
    
    logger.info("Fitting Gaussian scorer...")
    scorer = GaussianAnomalyScorer.fit(healthy_df)
    logger.info(f"Learned {len(scorer.sensor_cols)} sensors")
    
    logger.info("Scoring all training data...")
    scores = scorer.score(train_df)
    logger.info(f"Score range: {scores.min():.3f} to {scores.max():.3f}")
    
    anomalies = scorer.predict(train_df)
    logger.info(f"Flagged {anomalies.sum()} / {len(train_df)} as anomalous")

if __name__ == "__main__":
    main()