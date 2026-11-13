"""Run-directory, JSON/JSONL metadata and curve artifact writers."""
from __future__ import annotations
from pathlib import Path
import json, csv
from datetime import datetime, timezone

def make_run_dir(root: str, run_name: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    p = Path(root) / f"{stamp}_{run_name}"; p.mkdir(parents=True, exist_ok=False); return p

def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def append_ledger(path: str, row: dict) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f: f.write(json.dumps(row, sort_keys=True) + "\n")

def write_history(run_dir: Path, history: list[dict]) -> None:
    if not history: return
    with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=history[0].keys()); w.writeheader(); w.writerows(history)
    try:
        import matplotlib.pyplot as plt
        x=[r["epoch"] for r in history]; plt.figure(figsize=(8,4))
        plt.plot(x,[r["train_loss"] for r in history],label="train loss"); plt.plot(x,[r["val_loss"] for r in history],label="val loss")
        plt.plot(x,[r["val_macro_f1"] for r in history],label="val macro-F1"); plt.legend(); plt.xlabel("epoch"); plt.tight_layout(); plt.savefig(run_dir/"learning_curves.png",dpi=150); plt.close()
    except ImportError: pass
