"""Official BloodMNIST loading, auditing, and deterministic subset utilities.

This module deliberately preserves MedMNIST's provider-defined train/val/test
membership. It never creates a random validation or test split. Test samples are
available only when a caller explicitly requests ``split='test'``; the baseline
runner uses train and validation only.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from medmnist import BloodMNIST, INFO
from torch.utils.data import Dataset, Subset
from torchvision import transforms

SplitName = Literal["train", "val", "test"]

CLASS_NAMES: dict[int, str] = {
    int(index): label for index, label in INFO["bloodmnist"]["label"].items()
}


@dataclass(frozen=True)
class SplitProtocol:
    """Immutable identifiers for the official data and locked evaluation policy."""

    dataset_flag: str = "bloodmnist"
    dataset_version: str = "MedMNIST v2"
    source_size: int = 28
    model_size: int = 224
    train_size: int = 11959
    val_size: int = 1712
    test_size: int = 3421
    subset_fraction: float = 0.01
    subset_seed: int = 2026
    test_policy: str = "Locked: do not load labels or select models using test data."


def imagenet_eval_transform() -> transforms.Compose:
    """Return deterministic RGB resize/tensor/ImageNet-normalization transform."""
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
        ]
    )


def load_official_splits(
    data_root: str | Path,
    *,
    include_test: bool = False,
    download: bool = True,
    transform: transforms.Compose | None = None,
) -> dict[str, BloodMNIST]:
    """Load official provider splits with no resampling or re-partitioning.

    Parameters
    ----------
    data_root:
        Directory in which MedMNIST stores the 28px official archive.
    include_test:
        False by default to enforce the validation-only selection protocol.
    download:
        Pass through to MedMNIST. Colab may download the official archive.
    transform:
        Model transform. Defaults to deterministic 224px ImageNet preprocessing.

    Returns
    -------
    dict[str, BloodMNIST]
        ``train`` and ``val`` by default; explicit test opt-in adds ``test``.
    """
    transform = transform or imagenet_eval_transform()
    requested: tuple[SplitName, ...] = ("train", "val", "test") if include_test else ("train", "val")
    datasets = {
        split: BloodMNIST(split=split, root=str(data_root), size=28, download=download, transform=transform)
        for split in requested
    }
    protocol = SplitProtocol()
    expected = {"train": protocol.train_size, "val": protocol.val_size, "test": protocol.test_size}
    for split, dataset in datasets.items():
        if len(dataset) != expected[split]:
            raise ValueError(f"Unexpected {split} size {len(dataset)}; expected {expected[split]}.")
    return datasets


def labels_for(dataset: BloodMNIST) -> np.ndarray:
    """Return a one-dimensional integer label vector without accessing images."""
    return np.asarray(dataset.labels, dtype=np.int64).reshape(-1)


def class_count_rows(labels: np.ndarray) -> list[dict[str, int | float | str]]:
    """Calculate count and within-split prevalence for all eight classes."""
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    counts = np.bincount(labels, minlength=len(CLASS_NAMES))
    return [
        {"class_index": index, "class_name": CLASS_NAMES[index], "count": int(count),
         "prevalence": float(count / len(labels))}
        for index, count in enumerate(counts)
    ]


def stratified_indices(labels: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """Create deterministic proportional indices using largest-remainder allocation.

    Exactly ``round(n * fraction)`` examples are selected. Within every class,
    sampling is without replacement using NumPy's PCG64 default generator.
    """
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1].")
    target = round(len(labels) * fraction)
    classes, counts = np.unique(labels, return_counts=True)
    raw = counts * fraction
    allocation = np.floor(raw).astype(int)
    allocation[np.argsort(-(raw - allocation), kind="stable")[: target - allocation.sum()]] += 1
    rng = np.random.default_rng(seed)
    selected = [rng.choice(np.flatnonzero(labels == label), n, replace=False) for label, n in zip(classes, allocation, strict=True)]
    return np.sort(np.concatenate(selected)).astype(np.int64)


def fixed_subset(dataset: Dataset, indices_path: str | Path) -> Subset:
    """Load a committed subset manifest and return that exact training subset."""
    archive = np.load(indices_path)
    indices = np.asarray(archive["indices"], dtype=np.int64)
    if np.any(indices < 0) or np.any(indices >= len(dataset)):
        raise ValueError("Subset manifest contains indices outside the training split.")
    return Subset(dataset, indices.tolist())
