"""End-to-end, validation-only training orchestrator; callable from CLI or Colab."""
from __future__ import annotations
import random
import subprocess
import time
import numpy as np


def _seed(seed: int) -> None:
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def _gpu() -> str:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA T4 runtime required; refusing CPU execution")
    name = torch.cuda.get_device_name(0)
    if "T4" not in name:
        raise RuntimeError(f"T4 required, got {name}")
    return name


def _git_commit() -> str:
    """Return the checked-out source revision for run provenance."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def run_training(cfg: dict, provider_module: str) -> dict:
    """Execute an authorized train/validation-only run via the canonical provider.

    ``provider_module`` must export ``build_provider_datasets``. The provider is the
    sole archive/split owner and must return no test data.
    """
    import torch
    from torch.utils.data import DataLoader
    from .models import build_transforms, build_resnet18
    from .data import load_train_val
    from .engine import train_epoch, validate_epoch
    from .metrics import classification_report
    from .artifacts import make_run_dir, write_json, write_history, append_ledger
    from .checkpoints import save, load_for_resume
    from .config import config_fingerprint

    gpu = _gpu(); _seed(cfg["training"]["seed"]); started = time.time(); fp = config_fingerprint(cfg)
    run_dir = make_run_dir(cfg["artifacts"]["root"], cfg["run"]["name"])
    train_tf, val_tf = build_transforms()
    data = load_train_val(provider_module, cfg, train_tf, val_tf)
    batch_size = cfg["training"]["batch_size"]; workers = cfg["training"].get("num_workers", 2)
    train_dl = DataLoader(data.train, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_dl = DataLoader(data.val, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    device = torch.device("cuda"); model = build_resnet18().to(device); criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["training"]["epochs"])
    start, best = 0, -1.0; resume = cfg["training"].get("resume")
    if resume: start, best = load_for_resume(resume, fp, model, optimizer, scheduler)
    metadata = {"dataset": data.metadata, "config": cfg, "config_fingerprint": fp, "git_commit": _git_commit(), "gpu": gpu, "seed": cfg["training"]["seed"], "start_time_unix": started}
    write_json(run_dir / "run_metadata.json", metadata); history = []; best_report = None
    for epoch in range(start, cfg["training"]["epochs"]):
        train_metrics = train_epoch(model, train_dl, optimizer, criterion, device)
        val_metrics = validate_epoch(model, val_dl, criterion, device)
        report = classification_report(val_metrics.pop("labels"), val_metrics.pop("logits"))
        row = {"epoch": epoch + 1, "train_loss": train_metrics["loss"], "train_accuracy": train_metrics["accuracy"], "val_loss": val_metrics["loss"], "val_accuracy": report["accuracy"], "val_macro_f1": report["macro_f1"], "lr": optimizer.param_groups[0]["lr"]}
        history.append(row)
        payload = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "best_val_macro_f1": best, "config_fingerprint": fp}
        save(run_dir / "last.pt", payload)
        if report["macro_f1"] > best:
            best = report["macro_f1"]; best_report = report; payload["best_val_macro_f1"] = best
            save(run_dir / "best.pt", payload); write_json(run_dir / "best_validation_metrics.json", report)
        scheduler.step()
    write_history(run_dir, history)
    summary = {"run_dir": str(run_dir), "best_epoch": max(history, key=lambda row: row["val_macro_f1"])["epoch"], "best_validation": best_report, "runtime_seconds": time.time() - started, "gpu": gpu, "test_evaluated": False}
    write_json(run_dir / "summary.json", summary); append_ledger(cfg["artifacts"]["ledger_path"], summary)
    return summary
