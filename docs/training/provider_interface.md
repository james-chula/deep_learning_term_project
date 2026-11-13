# Canonical provider integration interface

The Data & Baseline package is the sole owner of archive access, audit validation, fixed split/subset membership, and application of the caller-supplied deterministic source-28px-to-224px transforms. The training package has no loader and must import the canonical **function** from the nominated baseline module:

```python
from bloodmnist_baselines import build_provider_datasets

result = build_provider_datasets(
    audit_manifest_path="artifacts/data_audit/audit_manifest.json",
    image_size=224,
    train_transform=train_transform,
    val_transform=val_transform,
    subset_manifest_path="splits/bloodmnist_train_1pct_seed2026.npz",  # or None
)
```

The exact public signature is:

```python
build_provider_datasets(
    *, audit_manifest_path, image_size=224, train_transform, val_transform,
    subset_manifest_path=None,
)
```

The return value may use any provider-owned dataclass, but must expose `.train`, `.val`, and `.metadata`. It must never load or return test data. `metadata` must include at least:

```python
{
  "split_counts": {"train": 11959, "val": 1712, "test": 3421},
  "archive_size": 28,
  "resize_to": 224,
  "test_policy": "locked_not_loaded",
  "audit_manifest_path": "artifacts/data_audit/audit_manifest.json",
  "subset_manifest_path": "splits/bloodmnist_train_1pct_seed2026.npz",  # or None
}
```

The training adapter validates these values after the call. It does not construct splits, modify data membership, or receive any test dataset.
