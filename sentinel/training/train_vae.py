import logging
from dataclasses import dataclass

import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from sentinel.models.vae import VAE, vae_loss

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    latent_dim: int = 16
    window_size: int = 30
    epochs: int = 50
    batch_size: int = 64
    lr: float = 1e-3
    seed: int = 42
    max_grad_norm: float = 1.0


def train(windows: np.ndarray, input_dim: int, cfg: TrainConfig) -> VAE:
    """Train a VAE on normalized sensor windows.

    Logs per-epoch train_loss to the active MLflow run if one is open;
    the caller is responsible for starting/ending the run context.
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    dataset = TensorDataset(torch.tensor(windows, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    model = VAE(input_dim=input_dim, latent_dim=cfg.latent_dim, window_size=cfg.window_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            recon, mu, log_var = model(batch)
            loss = vae_loss(batch, recon, mu, log_var)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.max_grad_norm)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        logger.info("Epoch %d/%d  loss=%.4f", epoch + 1, cfg.epochs, avg_loss)

        # Log to the caller's MLflow run if one is active — no coupling to run lifecycle.
        if mlflow.active_run():
            mlflow.log_metric("train_loss", avg_loss, step=epoch)

    return model
