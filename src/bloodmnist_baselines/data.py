"""Authoritative BloodMNIST train/validation provider.

This module deliberately constructs only the official train and validation
splits. There is no test-loading argument: validation workflows cannot request
or receive the locked test dataset through this public interface.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from hashlib import md5, sha256
from pathlib import Path
from typing import Any

import numpy as np
from medmnist import BloodMNIST, INFO
from torch.utils.data import Dataset, Subset

CLASS_NAMES = {int(index): label for index, label in INFO["bloodmnist"]["label"].items()}
EXPECTED_SPLIT_COUNTS = {"train": 11959, "val": 1712, "test": 3421}
EXPECTED_ARCHIVE_MD5 = "7053d0359d879ad8a5505303e11de1dc"


@dataclass(frozen=True)
class ProviderDatasets:
    """Official training/validation datasets and validated provenance metadata."""

    train: Dataset
    val: Dataset
    metadata: dict[str, Any]


def _sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _md5(path: str | Path) -> str:
    digest = md5()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_label(value: Any) -> str:
    """Normalize only insignificant class-label whitespace."""
    if not isinstance(value, str):
        raise ValueError("Audit manifest class labels must be strings.")
    normalized = re.sub(r"\s+", " ", value.strip())
    return re.sub(r"\s+\(", "(", normalized)


def _validate_class_mapping(raw_mapping: Any) -> None:
    if not isinstance(raw_mapping, dict):
        raise ValueError("Audit manifest class mapping must be an object.")
    try:
        observed = {int(key): _canonical_label(value) for key, value in raw_mapping.items()}
    except (TypeError, ValueError) as error:
        raise ValueError("Audit manifest class mapping is invalid.") from error
    expected = {key: _canonical_label(value) for key, value in CLASS_NAMES.items()}
    if set(observed) != set(expected):
        raise ValueError("Audit manifest class IDs differ from BloodMNIST.")
    if observed != expected:
        raise ValueError("Audit manifest class mapping differs from BloodMNIST.")


def _audit(path: str | Path) -> tuple[dict[str, Any], str]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Audit manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("dataset_flag") != "bloodmnist":
        raise ValueError("Audit manifest must identify BloodMNIST.")
    if manifest.get("official_28px_archive_md5") != EXPECTED_ARCHIVE_MD5:
        raise ValueError("Audit manifest does not match the official 28px archive.")
    if manifest.get("provider_split_sizes") != EXPECTED_SPLIT_COUNTS:
        raise ValueError("Audit manifest does not match official split counts.")
    _validate_class_mapping(manifest.get("class_mapping"))
    if not isinstance(manifest.get("source_local_sha256"), str):
        raise ValueError("Audit manifest is missing the recorded source SHA-256.")
    return manifest, _sha256(path)


def _fixed_train_subset(train: Dataset, subset_manifest_path: str | Path, audit: dict[str, Any]) -> tuple[Subset, str]:
    """Validate and apply the committed subset only to the official train set."""
    subset_path = Path(subset_manifest_path)
    if not subset_path.is_file():
        raise FileNotFoundError(f"Fixed subset manifest not found: {subset_path}")
    with np.load(subset_path) as saved:
        indices = np.asarray(saved["indices"], dtype=np.int64)
        if str(saved["source_split"]) != "train" or int(saved["source_size"]) != EXPECTED_SPLIT_COUNTS["train"]:
            raise ValueError("Subset manifest is not bound to the official training split.")
        if len(indices) != int(audit["subset_size"]) or len(np.unique(indices)) != len(indices):
            raise ValueError("Subset membership differs from the audited fixed subset.")
        if indices.size == 0 or indices.min() < 0 or indices.max() >= len(train):
            raise ValueError("Subset indices are outside official training bounds.")
    return Subset(train, indices.tolist()), _sha256(subset_path)


def build_provider_datasets(*, audit_manifest_path, image_size, train_transform, val_transform, subset_manifest_path=None) -> ProviderDatasets:
    """Return official train/validation data and provenance; never load test data.

    The caller supplies separate train and validation transforms, each responsible
    for the one 28px-to-224px resize. An optional committed subset manifest is
    applied only to training. The public keyword-only signature is the shared
    provider contract for all project workflows.
    """
    if image_size != 224:
        raise ValueError("BloodMNIST workflow requires image_size=224.")
    audit, audit_sha = _audit(audit_manifest_path)
    root = Path(os.environ.get("MEDMNIST_DATA_ROOT", ".data/medmnist"))
    # The provider must never construct the locked split.
    train: Dataset = BloodMNIST(split="train", root=str(root), size=28, download=True, transform=train_transform)
    val: Dataset = BloodMNIST(split="val", root=str(root), size=28, download=True, transform=val_transform)
    if len(train) != EXPECTED_SPLIT_COUNTS["train"] or len(val) != EXPECTED_SPLIT_COUNTS["val"]:
        raise ValueError("Provider returned unexpected official train/validation counts.")

    metadata: dict[str, Any] = {
        "dataset_flag": "bloodmnist",
        "archive_size": 28,
        "resize_to": 224,
        "split_counts": dict(EXPECTED_SPLIT_COUNTS),
        "test_policy": "locked_not_loaded",
        "audit_manifest_path": str(audit_manifest_path),
        "audit_manifest_sha256": audit_sha,
        "source_archive_sha256": audit["source_local_sha256"],
        "source_archive_md5": audit["official_28px_archive_md5"],
        "subset_manifest_path": None,
        "subset_manifest_sha256": None,
    }
    archive = root / "bloodmnist.npz"
    if archive.is_file():
        if _md5(archive) != EXPECTED_ARCHIVE_MD5:
            raise ValueError("Downloaded archive differs from official BloodMNIST MD5.")
        metadata["loaded_archive_sha256"] = _sha256(archive)
        metadata["loaded_archive_md5"] = EXPECTED_ARCHIVE_MD5

    if subset_manifest_path is not None:
        train, subset_sha = _fixed_train_subset(train, subset_manifest_path, audit)
        metadata["subset_manifest_path"] = str(subset_manifest_path)
        metadata["subset_manifest_sha256"] = subset_sha
    metadata["train_examples"] = len(train)
    metadata["validation_examples"] = len(val)
    return ProviderDatasets(train=train, val=val, metadata=metadata)
