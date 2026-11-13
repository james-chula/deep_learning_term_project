"""Run one approved BloodMNIST validation-only mode on Google Colab T4.

Run from the thin notebook only:
    PYTHONPATH=src python run_experiment.py

Sequence: T4 check -> seed -> train/validation data -> selected mode -> validate
-> save evidence. No test split is loaded, requested, or evaluated.
"""
import csv
import json
import random
import subprocess
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, roc_auc_score

from config import CONFIG
from data import get_data_loaders
from model import create_model, majority_probabilities
from train import train_one_epoch, validate, validate_majority


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


def package_versions():
    import matplotlib
    import medmnist
    import sklearn
    import torchvision
    return {
        "python": __import__("sys").version,
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "medmnist": medmnist.__version__, "scikit_learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__, "resnet18_weights": "IMAGENET1K_V1",
    }


def make_output_folder():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder = Path(CONFIG["output_root"]) / f"{stamp}_{CONFIG['run_name']}_{CONFIG['mode']}"
    folder.mkdir(parents=True, exist_ok=False)
    return folder


def save_json(path, payload):
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_checkpoint(path, payload):
    """Atomically replace a neural evidence checkpoint after an epoch completes."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def save_history(history, output_folder):
    if not history:
        return
    with (output_folder / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)


def save_plot(history, output_folder):
    if not history:
        return
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


def validation_metrics(labels, predictions, probabilities=None):
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    classes = list(range(CONFIG["num_classes"]))
    report = classification_report(labels, predictions, labels=classes, output_dict=True, zero_division=0)
    result = {
        "accuracy": float((labels == predictions).mean()),
        "macro_f1": float(__import__("sklearn.metrics", fromlist=["f1_score"]).f1_score(labels, predictions, labels=classes, average="macro", zero_division=0)),
        "per_class": [{"class_index": index, "support": int(report[str(index)]["support"]), "precision": float(report[str(index)]["precision"]), "recall": float(report[str(index)]["recall"]), "f1": float(report[str(index)]["f1-score"])} for index in classes],
    }
    if probabilities is not None:
        try:
            result["macro_ovr_roc_auc"] = float(roc_auc_score(labels, probabilities, labels=classes, multi_class="ovr", average="macro"))
        except ValueError:
            result["macro_ovr_roc_auc"] = None
    return result


def train_neural_mode(mode, train_loader, validation_loader, device, output_folder):
    model = create_model(CONFIG["num_classes"], mode).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    loss_function = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])
    best_f1, best_epoch, history = -1.0, None, []
    best_metrics = None
    for epoch in range(1, CONFIG["epochs"] + 1):
        train_result = train_one_epoch(model, train_loader, optimizer, loss_function, device)
        validation_result = validate(model, validation_loader, loss_function, device)
        row = {"epoch": epoch, "train_loss": train_result["loss"], "train_accuracy": train_result["accuracy"], "validation_loss": validation_result["loss"], "validation_accuracy": validation_result["accuracy"], "validation_macro_f1": validation_result["macro_f1"], "learning_rate": optimizer.param_groups[0]["lr"]}
        history.append(row)
        save_checkpoint(output_folder / "last_model.pt", {"epoch": epoch, "model_state_dict": model.state_dict()})
        if validation_result["macro_f1"] > best_f1:
            best_f1, best_epoch = validation_result["macro_f1"], epoch
            best_metrics = validation_result
            save_checkpoint(output_folder / "best_model.pt", {"epoch": epoch, "model_state_dict": model.state_dict()})
        scheduler.step()
    save_history(history, output_folder)
    save_plot(history, output_folder)
    return best_epoch, best_f1, validation_metrics(best_metrics["labels"], best_metrics["predictions"], best_metrics["probabilities"]), history


def main():
    mode = CONFIG["mode"]
    if mode not in {"majority", "linear_probe", "finetune"}:
        raise ValueError("CONFIG['mode'] must be majority, linear_probe, or finetune.")
    gpu_name = require_t4()
    set_seed(CONFIG["seed"])
    started = time.time()
    output_folder = make_output_folder()
    train_loader, validation_loader, provider_metadata = get_data_loaders(CONFIG)
    metadata = {"config": CONFIG, "provider": provider_metadata, "git_commit": current_commit(), "gpu": gpu_name, "package_versions": package_versions(), "test_evaluated": False, "test_policy": "locked_not_loaded"}
    save_json(output_folder / "run_metadata.json", metadata)
    save_json(output_folder / "config.json", CONFIG)

    if mode == "majority":
        train_labels = []
        for _, labels in train_loader:
            train_labels.extend(labels.view(-1).tolist())
        probabilities, majority_class = majority_probabilities(train_labels, len(validation_loader.dataset), CONFIG["num_classes"])
        result = validate_majority(probabilities, validation_loader)
        metrics = validation_metrics(result["labels"], result["predictions"], probabilities.numpy())
        summary = {"mode": mode, "majority_class": majority_class, "best_epoch": None, "best_validation_macro_f1": metrics["macro_f1"], "runtime_seconds": time.time() - started, "test_evaluated": False}
    else:
        best_epoch, best_f1, metrics, _ = train_neural_mode(mode, train_loader, validation_loader, torch.device("cuda"), output_folder)
        summary = {"mode": mode, "best_epoch": best_epoch, "best_validation_macro_f1": best_f1, "runtime_seconds": time.time() - started, "test_evaluated": False}
    save_json(output_folder / "metrics.json", metrics)
    save_json(output_folder / "summary.json", summary)
    print("Saved validation-only evidence to", output_folder)
    print(summary)


if __name__ == "__main__":
    main()
