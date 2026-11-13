"""Configuration loading and validation for validation-only ResNet-18 runs."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import hashlib, json
import yaml

REQUIRED = {"run", "data", "model", "training", "artifacts"}

def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML and reject unsafe/missing training-contract fields."""
    with Path(path).open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict) or REQUIRED - cfg.keys():
        raise ValueError(f"config must contain {sorted(REQUIRED)}")
    d, m, t = cfg["data"], cfg["model"], cfg["training"]
    if d.get("archive_size") != 28 or d.get("resize_to") != 224:
        raise ValueError("only audited 28px archive + deterministic 224 resize is permitted")
    if d.get("test_policy") != "locked_not_loaded":
        raise ValueError("test_policy must be locked_not_loaded")
    if m.get("name") != "resnet18" or not m.get("pretrained"):
        raise ValueError("first evidence run is pretrained ResNet-18")
    if t.get("selection_metric") != "val_macro_f1":
        raise ValueError("selection is validation macro-F1 only")
    if not isinstance(t.get("seed"), int) or t.get("epochs", 0) < 1:
        raise ValueError("training.seed and positive training.epochs required")
    return cfg

def config_fingerprint(cfg: dict[str, Any]) -> str:
    """Stable hash stored in checkpoints to prevent incompatible resume."""
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
