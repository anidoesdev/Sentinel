"""MIMII dataset loading for audio anomaly detection.

Supports two directory layouts — auto-detected from folder structure:

Layout A — "original" MIMII (pump, slider, valve from zenodo.org/record/3678171):
    data/mimii/{machine_type}/{machine_id}/normal/*.wav
    data/mimii/{machine_type}/{machine_id}/abnormal/*.wav
    Label comes from the subdirectory name. No pre-made train/test split;
    normal files are split 80/20 into train and test internally.

Layout B — "dcase" format (fan, DCASE 2020 Task 2):
    data/mimii/{machine_type}/train/normal_id_XX_*.wav
    data/mimii/{machine_type}/test/normal_id_XX_*.wav
    data/mimii/{machine_type}/test/anomaly_id_XX_*.wav
    Label comes from the filename prefix. Split is pre-made.

Both produce the same (train_dataset, test_dataset) output from make_datasets().
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as F_audio
from pydantic import BaseModel
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

MachineType = Literal["fan", "pump", "slider", "valve"]
Layout = Literal["original", "dcase", "auto"]


class MIMIIConfig(BaseModel):
    data_dir: Path = Path("data/mimii")
    machine_type: MachineType = "pump"
    # For "original" layout: filter to one machine ID (e.g. "id_00"). None = use all.
    # For "dcase" layout: filter by machine ID embedded in filename. None = use all.
    machine_id: str | None = "id_00"
    layout: Layout = "auto"
    train_split: float = 0.8   # used only for "original" layout (no pre-made split)
    sample_rate: int = 16_000
    segment_length_s: float = 10.0
    batch_size: int = 8
    num_workers: int = 0  # keep 0 on Windows to avoid DataLoader spawn issues

    model_config = {"arbitrary_types_allowed": True}


# ---------------------------------------------------------------------------
# Layout detection
# ---------------------------------------------------------------------------

def _detect_layout(machine_dir: Path) -> Literal["original", "dcase"]:
    """Infer layout from what subdirectories exist."""
    if (machine_dir / "train").exists() or (machine_dir / "test").exists():
        return "dcase"
    id_dirs = [d for d in machine_dir.iterdir() if d.is_dir() and d.name.startswith("id_")]
    if id_dirs:
        return "original"
    raise ValueError(
        f"Cannot detect MIMII layout in {machine_dir}.\n"
        f"Expected either 'train/'/'test/' subdirs (dcase) or 'id_XX/' subdirs (original)."
    )


# ---------------------------------------------------------------------------
# Path loading — original layout
# ---------------------------------------------------------------------------

def _load_paths_original(
    config: MIMIIConfig,
    machine_dir: Path,
    seed: int,
) -> tuple[list[Path], list[int], list[Path], list[int]]:
    """Return (train_paths, train_labels, test_paths, test_labels) for original layout.

    Collects normal + abnormal files from id_XX subdirectories, then splits
    normal files 80/20 into train (fit) and test (evaluate).
    All abnormal files go into test only — never seen at fit time.
    """
    id_dirs = sorted(d for d in machine_dir.iterdir() if d.is_dir() and d.name.startswith("id_"))

    if config.machine_id is not None:
        id_dirs = [d for d in id_dirs if d.name == config.machine_id]
        if not id_dirs:
            raise ValueError(
                f"machine_id='{config.machine_id}' not found in {machine_dir}. "
                f"Available: {[d.name for d in sorted(machine_dir.iterdir()) if d.is_dir()]}"
            )

    all_normal: list[Path] = []
    all_abnormal: list[Path] = []

    for id_dir in id_dirs:
        norm_dir = id_dir / "normal"
        abn_dir = id_dir / "abnormal"
        if norm_dir.exists():
            all_normal.extend(sorted(norm_dir.glob("*.wav")))
        if abn_dir.exists():
            all_abnormal.extend(sorted(abn_dir.glob("*.wav")))

    if not all_normal:
        raise ValueError(f"No normal WAV files found under {machine_dir}")

    # Split normal files into train / held-out-test
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(all_normal)).tolist()
    n_train = int(len(all_normal) * config.train_split)
    train_paths = [all_normal[i] for i in indices[:n_train]]
    val_normal_paths = [all_normal[i] for i in indices[n_train:]]

    test_paths = val_normal_paths + all_abnormal
    test_labels = [0] * len(val_normal_paths) + [1] * len(all_abnormal)

    logger.info(
        "Original layout — train: %d normal  |  test: %d normal + %d abnormal",
        len(train_paths), len(val_normal_paths), len(all_abnormal),
    )
    return train_paths, [0] * len(train_paths), test_paths, test_labels


# ---------------------------------------------------------------------------
# Path loading — dcase layout
# ---------------------------------------------------------------------------

def _parse_label_dcase(path: Path) -> int:
    """Parse label from filename prefix: 'normal_' → 0, 'anomaly_' → 1."""
    name = path.name.lower()
    if name.startswith("normal_"):
        return 0
    if name.startswith("anomaly_"):
        return 1
    raise ValueError(
        f"Cannot parse label from '{path.name}'. Expected 'normal_' or 'anomaly_' prefix."
    )


def _load_paths_dcase(
    config: MIMIIConfig,
    machine_dir: Path,
) -> tuple[list[Path], list[int], list[Path], list[int]]:
    """Return (train_paths, train_labels, test_paths, test_labels) for dcase layout."""
    train_dir = machine_dir / "train"
    test_dir = machine_dir / "test"

    train_paths = sorted(train_dir.glob("*.wav")) if train_dir.exists() else []
    test_paths = sorted(test_dir.glob("*.wav")) if test_dir.exists() else []

    if config.machine_id is not None:
        train_paths = [p for p in train_paths if f"_{config.machine_id}_" in p.name]
        test_paths = [p for p in test_paths if f"_{config.machine_id}_" in p.name]

    if not train_paths:
        raise ValueError(f"No train WAV files found in {train_dir}")
    if not test_paths:
        raise ValueError(f"No test WAV files found in {test_dir}")

    train_labels = [_parse_label_dcase(p) for p in train_paths]
    test_labels = [_parse_label_dcase(p) for p in test_paths]

    n_test_normal = sum(1 for l in test_labels if l == 0)
    n_test_abn = sum(1 for l in test_labels if l == 1)
    logger.info(
        "DCASE layout — train: %d normal  |  test: %d normal + %d anomalous",
        len(train_paths), n_test_normal, n_test_abn,
    )
    return train_paths, train_labels, test_paths, test_labels


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MIMIIAudioDataset(Dataset):
    """Loads MIMII WAV files and returns (waveform, label) pairs.

    Label: 0 = normal, 1 = anomalous.
    Waveform shape: [n_samples] — mono float32, resampled to sample_rate.
    All clips padded or trimmed to exactly segment_length_s seconds.
    """

    def __init__(
        self,
        paths: list[Path],
        labels: list[int],
        sample_rate: int = 16_000,
        segment_length_s: float = 10.0,
    ) -> None:
        if len(paths) != len(labels):
            raise ValueError("paths and labels must have the same length")
        self.paths = paths
        self.labels = labels
        self.sample_rate = sample_rate
        self.n_samples = int(segment_length_s * sample_rate)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        # soundfile returns [samples, channels] as float32 numpy array
        data, sr = sf.read(str(self.paths[idx]), dtype="float32", always_2d=True)
        waveform = torch.from_numpy(data.T)  # [channels, samples]

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)  # [n_samples]

        if sr != self.sample_rate:
            waveform = F_audio.resample(waveform, sr, self.sample_rate)

        if waveform.shape[0] < self.n_samples:
            waveform = torch.nn.functional.pad(waveform, (0, self.n_samples - waveform.shape[0]))
        else:
            waveform = waveform[: self.n_samples]

        return waveform, self.labels[idx]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_datasets(
    config: MIMIIConfig,
    seed: int = 42,
) -> tuple[MIMIIAudioDataset, MIMIIAudioDataset]:
    """Return (train_dataset, test_dataset) for the configured machine type.

    Auto-detects the directory layout unless config.layout is set explicitly.
    Train dataset contains only normal samples (for unsupervised fitting).
    Test dataset contains both normal and anomalous samples (for evaluation).
    """
    machine_dir = config.data_dir / config.machine_type
    if not machine_dir.exists():
        raise FileNotFoundError(
            f"Machine directory not found: {machine_dir}"
        )

    layout = config.layout if config.layout != "auto" else _detect_layout(machine_dir)
    logger.info("Detected layout: %s", layout)

    if layout == "original":
        tr_paths, tr_labels, te_paths, te_labels = _load_paths_original(
            config, machine_dir, seed
        )
    else:
        tr_paths, tr_labels, te_paths, te_labels = _load_paths_dcase(config, machine_dir)

    common = dict(sample_rate=config.sample_rate, segment_length_s=config.segment_length_s)
    return (
        MIMIIAudioDataset(tr_paths, tr_labels, **common),
        MIMIIAudioDataset(te_paths, te_labels, **common),
    )
