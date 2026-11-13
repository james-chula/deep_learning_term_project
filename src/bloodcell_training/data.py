"""Adapter for the canonical Data & Baseline provider.

This module never opens an archive, creates a split, resizes an image, or exposes a
 test dataset. It imports the provider function nominated by the integrated baseline
 package and validates only the train/validation contract it returns.
"""
from __future__ import annotations
from dataclasses import dataclass
from importlib import import_module
from typing import Any

EXPECTED_COUNTS = {"train": 11959, "val": 1712, "test": 3421}
CANONICAL_FUNCTION = "build_provider_datasets"


@dataclass(frozen=True)
class TrainValData:
    """Provider-neutral view of the canonical train/validation datasets."""

    train: Any
    val: Any
    metadata: dict[str, Any]


def resolve_provider_function(provider_module: str):
    """Import the canonical provider function from an integrated baseline module."""
    provider = getattr(import_module(provider_module), CANONICAL_FUNCTION, None)
    if not callable(provider):
        raise ImportError(
            f"{provider_module!r} must export callable {CANONICAL_FUNCTION}(...)"
        )
    return provider


def load_train_val(
    provider_module: str, cfg: dict[str, Any], train_transform: Any, val_transform: Any
) -> TrainValData:
    """Call the canonical provider and enforce train/validation-only invariants.

    The provider owns audit verification, fixed membership, source-28px loading and
    deterministic resize to 224px. Its public call must be
    ``build_provider_datasets(*, audit_manifest_path, image_size=224,
    train_transform, val_transform, subset_manifest_path=None)``.
    """
    d = cfg["data"]
    provider = resolve_provider_function(provider_module)
    result = provider(
        audit_manifest_path=d["audit_manifest_path"],
        image_size=d["resize_to"],
        train_transform=train_transform,
        val_transform=val_transform,
        subset_manifest_path=d.get("subset_manifest_path"),
    )
    try:
        result = TrainValData(result.train, result.val, dict(result.metadata))
    except AttributeError as exc:
        raise TypeError("canonical provider result must expose .train, .val, .metadata") from exc

    meta = result.metadata
    if meta.get("split_counts") != EXPECTED_COUNTS:
        raise ValueError("canonical provider reported unapproved provider split counts")
    if meta.get("archive_size") != 28 or meta.get("resize_to") != 224:
        raise ValueError("expected 28px source archive with deterministic 224 resize")
    if meta.get("test_policy") != "locked_not_loaded":
        raise ValueError("provider must affirm test is locked and not loaded")
    return result
