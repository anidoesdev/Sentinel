"""Audio anomaly scorer using frozen AST embeddings + PCA + Gaussian distance.

Architecture (inference path):
  raw waveform [n_samples]
    → AutoFeatureExtractor → mel-spectrogram [1024, 128]
    → frozen ASTModel → [CLS] embedding [768]
    → PCA (768 → n_components)
    → StandardScaler (unit variance per component)
    → squared L2 distance from origin = anomaly score

The squared L2 distance in standardized PCA space equals the squared Mahalanobis
distance with a diagonal covariance — numerically stable and interpretable.
Higher score = more anomalous.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor, ASTModel

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"


class ASTEncoder:
    """Frozen AST encoder: raw waveforms → 768-dim [CLS] embeddings.

    AST was pretrained on AudioSet (2M clips, 527 classes). We freeze all parameters
    and use the [CLS] token as a rich audio representation. No fine-tuning is needed —
    the pretrained embedding space already separates different types of sounds, which
    is all we need for unsupervised anomaly detection.
    """

    def __init__(self, device: str = "cpu", model_id: str = _DEFAULT_MODEL_ID) -> None:
        self.device = device
        self.sample_rate = 16_000  # AST was trained at 16kHz — do not change

        logger.info("Loading AST from %s (downloads ~330MB on first call)...", model_id)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        self.model = ASTModel.from_pretrained(model_id).to(device)
        self.model.eval()

        # Freeze every parameter — we are a feature extractor, not a trainer
        for param in self.model.parameters():
            param.requires_grad_(False)

        n_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        logger.info("AST loaded. %.1fM parameters frozen.", n_params)

    @torch.no_grad()
    def encode_batch(self, waveforms: torch.Tensor) -> np.ndarray:
        """Encode a batch of waveforms to [CLS] embeddings.

        Args:
            waveforms: [batch, n_samples] float32 tensor at 16kHz

        Returns:
            embeddings: [batch, 768] float32 numpy array
        """
        # HuggingFace feature extractor computes the mel-spectrogram with AST's
        # original parameters: 128 mel bins, 25ms window, 10ms hop, fmax=8000Hz.
        # It also handles mean/std normalization. Output shape: [batch, time, 128].
        waveforms_np = waveforms.cpu().numpy().tolist()
        inputs = self.feature_extractor(
            waveforms_np,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        # last_hidden_state: [batch, seq_len, 768]
        # seq_len = num_patches + 1 (the +1 is the [CLS] token at position 0)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [batch, 768]
        return cls_embeddings.cpu().numpy().astype(np.float32)

    def encode_dataset(self, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        """Encode all batches in a DataLoader. Returns (embeddings, labels)."""
        all_embeddings: list[np.ndarray] = []
        all_labels: list[int] = []

        for i, (waveforms, labels) in enumerate(loader):
            if i % 5 == 0:
                logger.info("  Encoding batch %d / %d...", i + 1, len(loader))
            all_embeddings.append(self.encode_batch(waveforms))
            all_labels.extend(labels.tolist())

        return np.concatenate(all_embeddings, axis=0), np.array(all_labels)


class AudioAnomalyScorer:
    """Unsupervised audio anomaly scorer: fit on normal sounds, score any sound.

    The anomaly score is the squared Mahalanobis distance of the AST embedding
    from the normal cluster center in PCA-reduced space. A score above threshold
    means the sound is anomalous.

    Fit-time steps (normal audio only):
      1. Extract AST embeddings for all normal training clips.
      2. Fit PCA (768 → n_components). This concentrates the normal cluster's
         variance and discards noise dimensions.
      3. Fit StandardScaler — zero mean, unit variance per component.
         After this, the normal cluster is a sphere at the origin.
      4. Compute training scores; set threshold at threshold_percentile.

    Score-time steps (any audio):
      1. Extract AST embeddings.
      2. Project with fitted PCA.
      3. Standardize with fitted scaler.
      4. Return sum of squared values per sample.
    """

    def __init__(
        self,
        encoder: ASTEncoder,
        pca: PCA,
        scaler: StandardScaler,
        threshold: float,
    ) -> None:
        self.encoder = encoder
        self.pca = pca
        self.scaler = scaler
        self.threshold = threshold

    @classmethod
    def fit(
        cls,
        train_loader: DataLoader,
        device: str = "cpu",
        n_pca_components: int = 32,
        threshold_percentile: float = 99.0,
        model_id: str = _DEFAULT_MODEL_ID,
    ) -> "AudioAnomalyScorer":
        """Fit the scorer on normal training audio.

        train_loader should only contain normal (label=0) samples.
        """
        encoder = ASTEncoder(device=device, model_id=model_id)

        logger.info("Step 1/3: Extracting AST embeddings from training data...")
        embeddings, _ = encoder.encode_dataset(train_loader)
        logger.info("Embeddings shape: %s", embeddings.shape)  # [n_files, 768]

        # Reduce dimensionality before fitting the Gaussian. 768 dims with a few
        # hundred training samples makes full covariance estimation ill-conditioned.
        # 32 PCA components typically explains >90% of normal-class variance.
        n_components = min(n_pca_components, embeddings.shape[0] - 1)
        logger.info("Step 2/3: Fitting PCA (%d → %d dims)...", embeddings.shape[1], n_components)
        pca = PCA(n_components=n_components, random_state=42)
        embeddings_pca = pca.fit_transform(embeddings)
        explained = pca.explained_variance_ratio_.sum()
        logger.info("PCA: %d components explain %.1f%% of variance.", n_components, explained * 100)

        logger.info("Step 3/3: Fitting scaler and calibrating threshold...")
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings_pca)

        # Squared L2 from origin = squared Mahalanobis distance (diagonal cov = identity)
        # Follows chi-squared(n_components) under the null (normal sound).
        train_scores = np.sum(embeddings_scaled**2, axis=1)
        threshold = float(np.percentile(train_scores, threshold_percentile))
        logger.info(
            "Threshold at %.0f-pctile of normal scores: %.2f  "
            "(mean=%.2f, std=%.2f, max=%.2f)",
            threshold_percentile,
            threshold,
            train_scores.mean(),
            train_scores.std(),
            train_scores.max(),
        )

        return cls(encoder=encoder, pca=pca, scaler=scaler, threshold=threshold)

    def score(self, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        """Return (anomaly_scores, true_labels) for all samples in loader."""
        embeddings, labels = self.encoder.encode_dataset(loader)
        embeddings_pca = self.pca.transform(embeddings)
        embeddings_scaled = self.scaler.transform(embeddings_pca)
        scores = np.sum(embeddings_scaled**2, axis=1)
        return scores, labels

    def predict(self, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        """Return (predicted_labels, true_labels). 1 = anomalous, 0 = normal."""
        scores, labels = self.score(loader)
        return (scores > self.threshold).astype(int), labels

    def save(self, path: Path) -> None:
        """Save scorer state (excludes the AST model — reloaded from HuggingFace Hub)."""
        import pickle

        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "pca": self.pca,
            "scaler": self.scaler,
            "threshold": self.threshold,
            "model_id": self.encoder.model.config._name_or_path,
            "device": self.encoder.device,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        logger.info("Saved AudioAnomalyScorer state to %s", path)

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "AudioAnomalyScorer":
        """Load a previously saved scorer."""
        import pickle

        with open(path, "rb") as f:
            state = pickle.load(f)
        encoder = ASTEncoder(device=device, model_id=state["model_id"])
        return cls(
            encoder=encoder,
            pca=state["pca"],
            scaler=state["scaler"],
            threshold=state["threshold"],
        )
