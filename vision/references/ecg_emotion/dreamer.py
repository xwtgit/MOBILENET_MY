import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset


CLASS_NAMES = ("LVLA", "LVHA", "HVLA", "HVHA")


@dataclass(frozen=True)
class DreamerSample:
    subject: int
    trial: int
    ecg: np.ndarray
    valence: float
    arousal: float
    label: int


def quadrant_label(valence: float, arousal: float, threshold: float = 3.0) -> int:
    low_v = valence < threshold
    low_a = arousal < threshold
    if low_v and low_a:
        return 0
    if low_v and not low_a:
        return 1
    if not low_v and low_a:
        return 2
    return 3


def _field(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj[name]
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, np.ndarray) and obj.dtype.names and name in obj.dtype.names:
        return obj[name]
    raise KeyError(name)


def _squeeze_obj(obj: Any) -> Any:
    while isinstance(obj, np.ndarray) and obj.dtype == object and obj.size == 1:
        obj = obj.item()
    return obj


def _as_sequence(obj: Any) -> list[Any]:
    obj = _squeeze_obj(obj)
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            return [_squeeze_obj(item) for item in obj.ravel()]
        if obj.ndim == 0:
            return [obj.item()]
        if obj.ndim == 1:
            return [item for item in obj]
        return [obj[i] for i in range(obj.shape[0])]
    if isinstance(obj, (list, tuple)):
        return list(obj)
    return [obj]


def _numeric_vector(obj: Any) -> np.ndarray:
    obj = _squeeze_obj(obj)
    arr = np.asarray(obj, dtype=np.float32).squeeze()
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr.reshape(-1)


def _orient_ecg(ecg: Any) -> np.ndarray:
    arr = np.asarray(_squeeze_obj(ecg), dtype=np.float32).squeeze()
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    if arr.ndim != 2:
        raise ValueError(f"Expected a 1D or 2D ECG array, got shape {arr.shape}")

    if arr.shape[0] <= 8 and arr.shape[1] > arr.shape[0]:
        return arr
    if arr.shape[1] <= 8 and arr.shape[0] > arr.shape[1]:
        return arr.T
    return arr


def _standardize(ecg: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = ecg.mean(axis=1, keepdims=True)
    std = ecg.std(axis=1, keepdims=True)
    return (ecg - mean) / np.maximum(std, eps)


def pad_or_crop(ecg: np.ndarray, max_length: int) -> np.ndarray:
    if ecg.shape[1] == max_length:
        return ecg
    if ecg.shape[1] > max_length:
        return ecg[:, :max_length]
    padded = np.zeros((ecg.shape[0], max_length), dtype=np.float32)
    padded[:, : ecg.shape[1]] = ecg
    return padded


def _root_struct(mat: dict[str, Any]) -> Any:
    if "DREAMER" in mat:
        return _squeeze_obj(mat["DREAMER"])
    user_keys = [key for key in mat if not key.startswith("__")]
    if len(user_keys) == 1:
        return _squeeze_obj(mat[user_keys[0]])
    raise KeyError("Could not find DREAMER root struct in .mat file")


def load_dreamer_samples(mat_path: str | Path, label_threshold: float = 3.0) -> list[DreamerSample]:
    mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    root = _root_struct(mat)
    data = _as_sequence(_field(root, "Data"))

    samples: list[DreamerSample] = []
    for subject_idx, subject in enumerate(data):
        subject = _squeeze_obj(subject)
        ecg_struct = _squeeze_obj(_field(subject, "ECG"))
        stimuli = _as_sequence(_field(ecg_struct, "stimuli"))
        valence = _numeric_vector(_field(subject, "ScoreValence"))
        arousal = _numeric_vector(_field(subject, "ScoreArousal"))

        trial_count = min(len(stimuli), len(valence), len(arousal))
        for trial_idx in range(trial_count):
            ecg = _orient_ecg(stimuli[trial_idx])
            v = float(valence[trial_idx])
            a = float(arousal[trial_idx])
            samples.append(
                DreamerSample(
                    subject=subject_idx + 1,
                    trial=trial_idx + 1,
                    ecg=ecg,
                    valence=v,
                    arousal=a,
                    label=quadrant_label(v, a, label_threshold),
                )
            )

    if not samples:
        raise RuntimeError("No DREAMER ECG samples were found")
    return samples


def _label_counts(samples: Iterable[DreamerSample]) -> Counter[int]:
    return Counter(sample.label for sample in samples)


def _score_split(counts: Counter[int], target: dict[int, float]) -> float:
    return sum(abs(counts.get(label, 0) - target.get(label, 0.0)) for label in range(len(CLASS_NAMES)))


def make_subject_split(
    samples: list[DreamerSample],
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
    seed: int = 0,
) -> dict[str, list[int]]:
    rng = np.random.default_rng(seed)
    by_subject: dict[int, list[DreamerSample]] = defaultdict(list)
    for sample in samples:
        by_subject[sample.subject].append(sample)

    subjects = list(by_subject)
    rng.shuffle(subjects)
    n_subjects = len(subjects)
    n_test = max(1, round(n_subjects * test_fraction))
    n_val = max(1, round(n_subjects * val_fraction))
    desired_sizes = {"test": n_test, "val": n_val, "train": n_subjects - n_test - n_val}

    total_counts = _label_counts(samples)
    split_subjects: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    split_counts: dict[str, Counter[int]] = {"train": Counter(), "val": Counter(), "test": Counter()}
    targets = {
        name: {label: total_counts[label] * desired_sizes[name] / n_subjects for label in range(len(CLASS_NAMES))}
        for name in split_subjects
    }

    for subject in subjects:
        subject_counts = _label_counts(by_subject[subject])
        candidates = [name for name, size in desired_sizes.items() if len(split_subjects[name]) < size]
        if not candidates:
            candidates = ["train"]
        best_split = min(
            candidates,
            key=lambda name: _score_split(split_counts[name] + subject_counts, targets[name]),
        )
        split_subjects[best_split].append(subject)
        split_counts[best_split].update(subject_counts)

    return {name: sorted(values) for name, values in split_subjects.items()}


def load_split(path: str | Path) -> dict[str, list[int]]:
    with open(path, "r", encoding="utf-8") as f:
        split = json.load(f)
    return {name: [int(subject) for subject in subjects] for name, subjects in split.items()}


def save_split(split: dict[str, list[int]], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2, sort_keys=True)


class DreamerECGDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self,
        mat_path: str | Path,
        split: str,
        split_json: str | Path | None = None,
        max_length: int | None = None,
        label_threshold: float = 3.0,
        seed: int = 0,
        val_fraction: float = 0.2,
        test_fraction: float = 0.2,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train, val, test")

        samples = load_dreamer_samples(mat_path, label_threshold)
        subject_split = load_split(split_json) if split_json else make_subject_split(samples, val_fraction, test_fraction, seed)
        subject_ids = set(subject_split[split])
        selected = [sample for sample in samples if sample.subject in subject_ids]
        if not selected:
            raise RuntimeError(f"No samples selected for split {split}")

        if max_length is None:
            max_length = max(sample.ecg.shape[1] for sample in samples)
        self.max_length = int(max_length)
        self.split = split
        self.subjects = sorted(subject_ids)
        self.samples = selected

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[index]
        ecg = _standardize(sample.ecg.astype(np.float32, copy=False))
        ecg = pad_or_crop(ecg, self.max_length)
        return torch.from_numpy(ecg), sample.label

    def label_counts(self) -> dict[str, int]:
        counts = _label_counts(self.samples)
        return {CLASS_NAMES[label]: counts.get(label, 0) for label in range(len(CLASS_NAMES))}
