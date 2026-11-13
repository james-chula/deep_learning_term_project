"""Run one approved, validation-only BloodMNIST ResNet-18 experiment.

Run only from Google Colab with an NVIDIA T4 GPU:
    PYTHONPATH=src python run_experiment.py

The test split is deliberately unavailable in this script and in data.py.
"""
import csv
import json
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from config import CONFIG
from data import get_data_loaders
from model import create_model
from train import train_one_epoch, validate


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def require_t4():
    if not torch.cuda.is_available():
        raise RuntimeError("Use Google Colab with an NVIDIA T4 GPU; CPU runs are not allowed.")
    gpu_name = torch.cuda.get_device_name(0)
    if "T4" not in gpu_name:
        raise RuntimeError(f"An NVIDIA T4 GPU is required; found: {gpu_name}")
    return gpu_name


def current_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def make_output_folder():
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder = Path(CONFIG["output_root"]) / f"{timestamp}_{CONFIG['run_name']}"
    folder.mkdir(parents=True, exist_ok=False)
    return folder


def save_history(history, output_folder):
    with (output_folder / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)


def save_plot(history, output_folder):
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, [row["train_loss"] for row in history], label="train loss")
    plt.plot(epochs, [row["validation_loss"] for row in history], label="validation loss")
    plt.plot(epochs, [row["validation_macro_f1"] for row in history], label="validation macro-F1")
    plt.xlabel("epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_folder / "learning_curves.png", dpi=150)
    plt.close()


def main():
    gpu_name = require_t4()
    set_seed(CONFIG["seed"])
    started = time.time()
    output_folder = make_output_folder()

    train_loader, validation_loader, provider_metadata = get_data_loaders(CONFIG)
    device = torch.device("cuda")
    model = create_model(CONFIG["num_classes"]).to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])

    metadata = {
        "config": CONFIG,
        "provider": provider_metadata,
        "git_commit": current_commit(),
        "gpu": gpu_name,
        "test_evaluated": False,
    }
    (output_folder / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    best_macro_f1 = -1.0
    best_epoch = None
    history = []
    for epoch in range(1, CONFIG["epochs"] + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, loss_function, device)
        validation_metrics = validate(model, validation_loader, loss_function, device)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "validation_loss": validation_metrics["loss"],
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_macro_f1": validation_metrics["macro_f1"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(row)

        torch.save({"epoch": epoch, "model_state_dict": model.state_dict()}, output_folder / "last_model.pt")
        if validation_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = validation_metrics["macro_f1"]
            best_epoch = epoch
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict()}, output_folder / "best_model.pt")
        scheduler.step()

    save_history(history, output_folder)
    save_plot(history, output_folder)
    summary = {
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_macro_f1,
        "runtime_seconds": time.time() - started,
        "test_evaluated": False,
    }
    (output_folder / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Saved results to", output_folder)
    print(summary)


if __name__ == "__main__":
    main()
