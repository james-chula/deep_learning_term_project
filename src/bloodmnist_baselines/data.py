"""Authoritative BloodMNIST train/validation provider.

This provider constructs only the official train and validation splits. It has no
test-loading option, so downstream validation workflows cannot accidentally load
the locked test set.
"""
import json
import os
from hashlib import md5, sha256
from pathlib import Path

import numpy as np
from medmnist import BloodMNIST, INFO

CLASS_NAMES = {int(index): label for index, label in INFO["bloodmnist"]["label"].items()}
EXPECTED_SPLIT_COUNTS = {"train": 11959, "val": 1712, "test": 3421}
EXPECTED_ARCHIVE_MD5 = "7053d0359d879ad8a5505303e11de1dc"


def _sha256(path):
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _md5(path):
    digest = md5()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _audit(path):
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
    if {int(key): value for key, value in manifest.get("class_mapping", {}).items()} != CLASS_NAMES:
        raise ValueError("Audit manifest class mapping differs from BloodMNIST.")
    return manifest, _sha256(path)


def build_provider_datasets(*, audit_manifest_path, image_size, train_transform, val_transform, subset_manifest_path=None):
    """Return only official train/validation datasets plus reproducibility metadata.

    Image transforms supplied by the caller perform the single 28px-to-224px
    resize. An explicitly requested fixed subset is validated and applied only to
    the official training dataset; no subset path means full official training.
    """
    if image_size != 224:
        raise ValueError("BloodMNIST workflow requires image_size=224.")
    audit, audit_sha = _audit(audit_manifest_path)
    root = Path(os.environ.get("MEDMNIST_DATA_ROOT", ".data/medmnist"))
    train = BloodMNIST(split="train", root=str(root), size=28, download=True, transform=train_transform)
    val = BloodMNIST(split="val", root=str(root), size=28, download=True, transform=val_transform)
    if len(train) != EXPECTED_SPLIT_COUNTS["train"] or len(val) != EXPECTED_SPLIT_COUNTS["val"]:
        raise ValueError("Provider returned unexpected official train/validation counts.")
    metadata = {"dataset_flag": "bloodmnist", "archive_size": 28, "resize_to": 224, "split_counts": dict(EXPECTED_SPLIT_COUNTS), "test_policy": "locked_not_loaded", "audit_manifest_path": str(audit_manifest_path), "audit_manifest_sha256": audit_sha, "source_archive_sha256": audit["source_local_sha256"], "source_archive_md5": audit["official_28px_archive_md5"], "subset_manifest_path": None, "subset_manifest_sha256": None}
    archive = root / "bloodmnist.npz"
    if archive.is_file():
        if _md5(archive) != EXPECTED_ARCHIVE_MD5:
            raise ValueError("Downloaded archive differs from official BloodMNIST MD5.")
        metadata["loaded_archive_sha256"] = _sha256(archive)
        metadata["loaded_archive_md5"] = EXPECTED_ARCHIVE_MD5
    if subset_manifest_path is not None:
        subset_path = Path(subset_manifest_path)
        if not subset_path.is_file():
            raise FileNotFoundError(f"Fixed subset manifest not found: {subset_path}")
        with np.load(subset_path) as saved:
            indices = np.asarray(saved["indices"], dtype=np.int64)
            if str(saved["source_split"]) != "train" or int(saved["source_size"]) != EXPECTED_SPLIT_COUNTS["train"]:
                raise ValueError("Subset manifest is not bound to the official training split.")
            if len(indices) != int(audit["subset_size"]) or len(np.unique(indices)) != len(indices):
                raise ValueError("Subset membership differs from the audited fixed subset.")
            if indices.min() < 0 or indices.max() >= len(train):
                raise ValueError("Subset indices are outside official training bounds.")
        from torch.utils.data import Subset
        train = Subset(train, indices.tolist())
        metadata["subset_manifest_path"] = str(subset_path)
        metadata["subset_manifest_sha256"] = _sha256(subset_path)
        metadata["train_examples"] = len(train)
    else:
        metadata["train_examples"] = len(train)
    metadata["validation_examples"] = len(val)
    return {"train": train, "val": val, "metadata": metadata}
