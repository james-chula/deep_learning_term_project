"""Official BloodMNIST loading, auditing, and deterministic subset utilities.

This module deliberately preserves MedMNIST's provider-defined train/validation
membership.  The training adapter never requests or returns the test split.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import md5, sha256
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
from medmnist import BloodMNIST, INFO
from torch.utils.data import Dataset, Subset
from torchvision import transforms

SplitName = Literal["train", "val", "test"]

CLASS_NAMES: dict[int, str] = {
    int(index): label for index, label in INFO["bloodmnist"]["label"].items()
}
_EXPECTED_SPLIT_COUNTS = {"train": 11959, "val": 1712, "test": 3421}
_EXPECTED_ARCHIVE_MD5 = "7053d0359d879ad8a5505303e11de1dc"


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


@dataclass(frozen=True)
class ProviderDatasets:
    """Train/validation datasets and immutable provenance for downstream training.

    ``train`` can be a :class:`~torch.utils.data.Subset` only when the caller
    explicitly supplies the tracked fixed-subset manifest. ``val`` is always the
    complete provider validation split.  Test data are intentionally absent.
    """

    train: Dataset
    val: BloodMNIST
    metadata: dict[str, Any]


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_md5(path: Path) -> str:
    digest = md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_and_validate_audit_manifest(path: str | Path) -> tuple[dict[str, Any], str]:
    """Read the tracked audit JSON and reject a non-BloodMNIST split contract."""
    audit_path = Path(path)
    if not audit_path.is_file():
        raise FileNotFoundError(f"Audit manifest not found: {audit_path}")
    with audit_path.open(encoding="utf-8") as handle:
        audit = json.load(handle)
    if audit.get("dataset_flag") != "bloodmnist":
        raise ValueError("Audit manifest must identify dataset_flag='bloodmnist'.")
    if audit.get("official_28px_archive_md5") != _EXPECTED_ARCHIVE_MD5:
        raise ValueError("Audit manifest does not contain the expected official 28px archive MD5.")
    if audit.get("provider_split_sizes") != _EXPECTED_SPLIT_COUNTS:
        raise ValueError("Audit manifest provider split counts differ from the fixed BloodMNIST contract.")
    class_mapping = {int(key): value for key, value in audit.get("class_mapping", {}).items()}
    if class_mapping != CLASS_NAMES:
        raise ValueError("Audit manifest class mapping differs from MedMNIST BloodMNIST metadata.")
    source_sha = audit.get("source_local_sha256")
    if not isinstance(source_sha, str) or len(source_sha) != 64:
        raise ValueError("Audit manifest must contain a 64-character source_local_sha256 identity.")
    return audit, _file_sha256(audit_path)


def imagenet_eval_transform() -> transforms.Compose:
    """Return deterministic RGB resize/tensor/ImageNet-normalization transform."""
    return transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def load_official_splits(
    data_root: str | Path, *, include_test: bool = False, download: bool = True,
    transform: transforms.Compose | None = None,
) -> dict[str, BloodMNIST]:
    """Load official provider splits with no resampling or re-partitioning.

    ``include_test`` exists only for non-selection audit use. The training adapter
    below always calls this function with its default false value.
    """
    transform = transform or imagenet_eval_transform()
    requested: tuple[SplitName, ...] = ("train", "val", "test") if include_test else ("train", "val")
    datasets = {split: BloodMNIST(split=split, root=str(data_root), size=28,
                                  download=download, transform=transform) for split in requested}
    for split, dataset in datasets.items():
        if len(dataset) != _EXPECTED_SPLIT_COUNTS[split]:
            raise ValueError(f"Unexpected {split} size {len(dataset)}; expected {_EXPECTED_SPLIT_COUNTS[split]}.")
    return datasets


def build_provider_datasets(
    *, audit_manifest_path: str | Path, image_size: int,
    train_transform: Any, val_transform: Any,
    subset_manifest_path: str | Path | None = None,
) -> ProviderDatasets:
    """Build the locked provider train/validation contract for training pipelines.

    Parameters are keyword-only so a pipeline cannot accidentally request test
    data. ``image_size`` is validated as 224; resizing itself belongs to the
    supplied transforms. The root is obtained from ``MEDMNIST_DATA_ROOT`` or
    defaults to ``.data/medmnist``. If ``subset_manifest_path`` is supplied it
    is validated against the audit contract and applied **only** to train.

    The returned metadata carries SHA-256 identities for the audit and optional
    subset manifest, plus the audit-recorded source archive identity. If a local
    28px archive is present, actual SHA-256 and official MD5 are also recorded;
    a wrong MD5 is rejected. No test dataset is constructed or returned.
    """
    if image_size != 224:
        raise ValueError("BloodMNIST transfer-learning contract requires image_size=224.")
    audit, audit_sha = _read_and_validate_audit_manifest(audit_manifest_path)
    import os
    data_root = Path(os.environ.get("MEDMNIST_DATA_ROOT", ".data/medmnist"))
    # Explicitly false: test labels are neither loaded nor exposed through this API.
    datasets = load_official_splits(data_root, include_test=False, download=True, transform=train_transform)
    train, val = datasets["train"], BloodMNIST(split="val", root=str(data_root), size=28,
                                                download=True, transform=val_transform)
    if len(val) != _EXPECTED_SPLIT_COUNTS["val"]:
        raise ValueError("Unexpected provider validation split size.")

    metadata: dict[str, Any] = {
        "dataset_flag": "bloodmnist", "archive_size": 28, "resize_to": 224,
        "split_counts": dict(_EXPECTED_SPLIT_COUNTS), "test_policy": "locked_not_loaded",
        "audit_manifest_path": str(audit_manifest_path), "audit_manifest_sha256": audit_sha,
        "source_archive_sha256": audit["source_local_sha256"],
        "source_archive_md5": audit["official_28px_archive_md5"],
        "subset_manifest_path": None, "subset_manifest_sha256": None,
    }
    archive_path = data_root / "bloodmnist.npz"
    if archive_path.is_file():
        actual_md5 = _file_md5(archive_path)
        if actual_md5 != _EXPECTED_ARCHIVE_MD5:
            raise ValueError("Downloaded BloodMNIST archive MD5 differs from the official audit identity.")
        metadata["loaded_archive_sha256"] = _file_sha256(archive_path)
        metadata["loaded_archive_md5"] = actual_md5

    if subset_manifest_path is not None:
        subset_path = Path(subset_manifest_path)
        if not subset_path.is_file():
            raise FileNotFoundError(f"Subset manifest not found: {subset_path}")
        with np.load(subset_path) as subset_archive:
            indices = np.asarray(subset_archive["indices"], dtype=np.int64)
            if (int(subset_archive["source_size"]) != _EXPECTED_SPLIT_COUNTS["train"]
                    or str(subset_archive["source_split"]) != "train"):
                raise ValueError("Subset manifest is not bound to the official train split.")
            if len(indices) != int(audit["subset_size"]) or len(np.unique(indices)) != len(indices):
                raise ValueError("Subset manifest does not match audited fixed-subset membership.")
        train = fixed_subset(train, subset_path)
        metadata["subset_manifest_path"] = str(subset_path)
        metadata["subset_manifest_sha256"] = _file_sha256(subset_path)
        metadata["train_examples"] = len(train)
    else:
        metadata["train_examples"] = len(train)
    metadata["validation_examples"] = len(val)
    return ProviderDatasets(train=train, val=val, metadata=metadata)


def labels_for(dataset: BloodMNIST) -> np.ndarray:
    """Return a one-dimensional integer label vector without accessing images."""
    return np.asarray(dataset.labels, dtype=np.int64).reshape(-1)


def class_count_rows(labels: np.ndarray) -> list[dict[str, int | float | str]]:
    """Calculate count and within-split prevalence for all eight classes."""
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    counts = np.bincount(labels, minlength=len(CLASS_NAMES))
    return [{"class_index": index, "class_name": CLASS_NAMES[index], "count": int(count),
             "prevalence": float(count / len(labels))} for index, count in enumerate(counts)]


def stratified_indices(labels: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """Create deterministic proportional indices using largest-remainder allocation."""
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1].")
    target = round(len(labels) * fraction)
    classes, counts = np.unique(labels, return_counts=True)
    raw = counts * fraction
    allocation = np.floor(raw).astype(int)
    allocation[np.argsort(-(raw - allocation), kind="stable")[: target - allocation.sum()]] += 1
    rng = np.random.default_rng(seed)
    selected = [rng.choice(np.flatnonzero(labels == label), n, replace=False)
                for label, n in zip(classes, allocation, strict=True)]
    return np.sort(np.concatenate(selected)).astype(np.int64)


def fixed_subset(dataset: Dataset, indices_path: str | Path) -> Subset:
    """Load a committed subset manifest and return that exact training subset."""
    with np.load(indices_path) as archive:
        indices = np.asarray(archive["indices"], dtype=np.int64)
    if np.any(indices < 0) or np.any(indices >= len(dataset)):
        raise ValueError("Subset manifest contains indices outside the training split.")
    return Subset(dataset, indices.tolist())
