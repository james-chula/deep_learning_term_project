"""Atomic checkpoint save/load with config-fingerprint resume protection."""
from __future__ import annotations
from pathlib import Path
import os, torch

def save(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp"); torch.save(payload, tmp); os.replace(tmp, path)

def load_for_resume(path: str, fingerprint: str, model, optimizer, scheduler):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("config_fingerprint") != fingerprint: raise ValueError("resume checkpoint config fingerprint mismatch")
    model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"])
    return int(state["epoch"])+1, float(state["best_val_macro_f1"])
