import logging
from pathlib import Path
import pandas as pd
from sentinel.data.cmapss import load_raw,drop_low_variance,get_healthy_cycle, CMAPSSConfig
from sentinel.models.statistical_baseline import GaussianBaseline
from sklearn.metrics import classification_report

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
    baseline = GaussianBaseline()
    sensor_cols = [col for col in [f"sensor_{i}" for i in range(1,22)] if col not in dropped]
    baseline.fit(healthy_df,sensor_cols)
    logger.info(f"Learned {len(baseline.sensor_cols_)} sensors")
    
    test_df = load_raw(config=config,split="test")
    test_df = test_df.drop(columns=dropped)
    
    

    score_test = baseline.score(test_df,sensor_cols)
    score_per_unit = test_df.assign(score=score_test).groupby("unit")["score"].max()
    train_scores = baseline.score(healthy_df, sensor_cols)
    threshold = baseline.threshold(train_scores)
    
    rul_df = pd.read_csv(config.data_dir/"raw"/f"RUL_FD00{config.fd_id}.txt",header=None,names=["RUL"])
    true_labels = (rul_df["RUL"] <= 30).astype(int).values
    pred_lables = (score_per_unit.values > threshold).astype(int)
    
    print(classification_report(true_labels,pred_lables))
    

if __name__ == "__main__":
    main()
    