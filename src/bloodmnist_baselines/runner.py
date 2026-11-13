"""Single validation-only entry point for the project's mandatory baselines.

Invoke from the thin Colab notebook, for example:
``python -m bloodmnist_baselines.runner --config configs/baseline_validation.yaml``.
The command intentionally has no test-evaluation flag.
"""
from __future__ import annotations

import argparse
import json
import platform
import random
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
import yaml

from .baselines import LinearProbeConfig, frozen_resnet18_linear_probe, majority_class_baseline
from .data import SplitProtocol, fixed_subset, labels_for, load_official_splits
from .metrics import metric_payload


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def run_validation_baselines(config: dict[str, Any], source_commit: str) -> dict[str, Any]:
    """Run required baselines on official validation only and save structured evidence."""
    if not __import__("re").fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("source_commit must be a full 40-character lowercase Git SHA.")
    if config.get("evaluation_split") != "val":
        raise ValueError("This entry point only permits validation evaluation.")
    if config.get("run_test_evaluation", False):
        raise ValueError("Test evaluation is prohibited until validation decisions are locked.")
    seed = int(config["seed"])
    _seed_everything(seed)
    started = perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.get("require_cuda", True) and device.type != "cuda":
        raise RuntimeError("This project requires Google Colab T4 for experiments; CUDA was not available.")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else None
    if config.get("require_t4", True) and (gpu_name is None or "T4" not in gpu_name.upper()):
        raise RuntimeError(f"This project requires a Colab T4 GPU; detected {gpu_name!r}.")
    datasets = load_official_splits(config["data_root"], include_test=False, download=True)
    train = datasets["train"]
    if config.get("train_subset") == "1pct":
        train = fixed_subset(train, config["subset_manifest"])
    elif config.get("train_subset") != "100pct":
        raise ValueError("train_subset must be '1pct' or '100pct'.")
    val = datasets["val"]
    train_labels = labels_for(train.dataset)[np.asarray(train.indices)] if hasattr(train, "indices") else labels_for(train)
    val_labels = labels_for(val)
    results: dict[str, Any] = {
        "status": "complete", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "split_protocol": SplitProtocol().__dict__, "evaluation_split": "val", "train_subset": config["train_subset"],
        "seed": seed, "source_commit": source_commit, "device": str(device), "gpu_name": gpu_name, "torch_version": torch.__version__,
        "python_version": platform.python_version(), "cuda_version": torch.version.cuda,
        "train_examples": len(train), "validation_examples": len(val),
    }
    majority_p, majority_meta = majority_class_baseline(train_labels, val_labels)
    results["majority_class"] = {**majority_meta, "metrics": metric_payload(val_labels, majority_p)}
    probe_config = LinearProbeConfig(batch_size=int(config["batch_size"]), num_workers=int(config["num_workers"]), seed=seed,
                                     logistic_c=float(config["linear_probe"]["C"]), max_iter=int(config["linear_probe"]["max_iter"]))
    probe_p, probe_meta = frozen_resnet18_linear_probe(train, val, device, probe_config)
    results["frozen_resnet18_linear_probe"] = {**probe_meta, "metrics": metric_payload(val_labels, probe_p)}
    results["total_runtime_seconds"] = perf_counter() - started
    output = Path(config["output_dir"]) / f"baseline_validation_{config['train_subset']}_seed{seed}.json"
    _write_json(output, results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BloodMNIST mandatory validation baselines on Colab T4.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-commit", required=True, help="Exact checked-out 40-character Git SHA.")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    results = run_validation_baselines(config, args.source_commit)
    print(json.dumps({"output_dir": config["output_dir"], "macro_f1": {name: item["metrics"]["macro_f1"] for name, item in results.items() if isinstance(item, dict) and "metrics" in item}}, indent=2))


if __name__ == "__main__":
    main()
