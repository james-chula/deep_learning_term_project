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
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

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


def class_names_from_audit(audit_manifest_path):
    """Return class names ordered by integer class ID from the audited manifest."""
    manifest = json.loads(Path(audit_manifest_path).read_text(encoding="utf-8"))
    mapping = manifest.get("class_mapping")
    if not isinstance(mapping, dict):
        raise ValueError("Audit manifest is missing class_mapping.")
    expected = list(range(CONFIG["num_classes"]))
    observed = sorted(int(key) for key in mapping)
    if observed != expected:
        raise ValueError("Audit class IDs do not match configured classes.")
    return [mapping[str(index)] for index in expected]


def display_class_name(class_id, class_names):
    """Return a concise plot label while preserving canonical class names elsewhere."""
    aliases = {
        3: "immature granulocytes",
    }
    class_id = int(class_id)
    return aliases.get(class_id, class_names[class_id])


def evaluate_best_checkpoint(mode, validation_loader, device, output_folder):
    """Reload and evaluate the macro-F1-selected neural checkpoint on validation."""
    if mode not in {"linear_probe", "finetune"}:
        raise ValueError("Best-checkpoint evaluation applies only to neural modes.")
    checkpoint_path = Path(output_folder) / "best_model.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Selected checkpoint not found: {checkpoint_path}")
    model = create_model(CONFIG["num_classes"], mode).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    result = validate(model, validation_loader, torch.nn.CrossEntropyLoss(), device)
    return model, checkpoint["epoch"], result


def save_confusion_matrices(labels, predictions, class_names, output_folder):
    """Save count and row-normalized validation confusion matrices."""
    import matplotlib.pyplot as plt
    classes = list(range(len(class_names)))
    counts = confusion_matrix(labels, predictions, labels=classes)
    row_totals = counts.sum(axis=1, keepdims=True)
    normalized = np.divide(counts, row_totals, out=np.zeros_like(counts, dtype=float), where=row_totals != 0)
    for matrix, title, filename, fmt in [
        (counts, "Validation confusion matrix (counts)", "confusion_matrix_counts.png", "d"),
        (normalized, "Validation confusion matrix (row-normalized)", "confusion_matrix_normalized.png", ".2f"),
    ]:
        figure, axis = plt.subplots(figsize=(10, 8))
        image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=None if fmt == "d" else 1)
        axis.set(xticks=classes, yticks=classes, xticklabels=class_names, yticklabels=class_names,
                 xlabel="Predicted class", ylabel="True class", title=title)
        plt.setp(axis.get_xticklabels(), rotation=45, ha="right")
        threshold = float(matrix.max()) / 2 if matrix.size else 0
        for row in classes:
            for column in classes:
                value = format(matrix[row, column], fmt)
                axis.text(column, row, value, ha="center", va="center",
                          color="white" if matrix[row, column] > threshold else "black", fontsize=8)
        figure.colorbar(image, ax=axis)
        figure.tight_layout()
        figure.savefig(Path(output_folder) / filename, dpi=150)
        plt.close(figure)


def save_per_class_chart(metrics, class_names, output_folder):
    """Save grouped validation precision/recall/F1 bars."""
    import matplotlib.pyplot as plt
    rows = metrics["per_class"]
    positions = np.arange(len(rows))
    width = 0.25
    figure, axis = plt.subplots(figsize=(12, 5))
    for offset, key, label in [(-width, "precision", "Precision"), (0, "recall", "Recall"), (width, "f1", "F1")]:
        axis.bar(positions + offset, [row[key] for row in rows], width, label=label)
    axis.set(ylim=(0, 1), ylabel="Score", title="Selected-checkpoint validation metrics by class")
    axis.set_xticks(positions, class_names, rotation=45, ha="right")
    axis.legend()
    figure.tight_layout()
    figure.savefig(Path(output_folder) / "per_class_metrics.png", dpi=150)
    plt.close(figure)


def save_misclassified_samples(validation_loader, predictions, class_names, output_folder, max_items=16):
    """Save the first validation mistakes in loader order without touching test data."""
    import matplotlib.pyplot as plt
    mean = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
    prediction_index = 0
    mistakes = []
    for images, labels in validation_loader:
        for image, label in zip(images, labels.view(-1)):
            predicted = int(predictions[prediction_index])
            prediction_index += 1
            if predicted != int(label) and len(mistakes) < max_items:
                mistakes.append((image.cpu(), int(label), predicted))
        if len(mistakes) >= max_items:
            break
    if prediction_index > len(predictions):
        raise ValueError("Prediction count is shorter than validation data.")
    if not mistakes:
        return None
    columns = 4
    rows = int(np.ceil(len(mistakes) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(12, 3 * rows), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for axis, (image, truth, predicted) in zip(axes.flat, mistakes):
        display = (image * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        axis.imshow(display)
        axis.set_title(f"true: {class_names[truth]}\npred: {class_names[predicted]}", fontsize=8)
        axis.axis("off")
    figure.suptitle("Misclassified validation samples")
    figure.tight_layout()
    destination = Path(output_folder) / "misclassified_samples.png"
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return destination


def save_validation_diagnostics(labels, predictions, metrics, validation_loader, class_names, output_folder):
    """Write all validation-only diagnostic figures for the selected result."""
    save_confusion_matrices(labels, predictions, class_names, output_folder)
    save_per_class_chart(metrics, class_names, output_folder)
    save_misclassified_samples(validation_loader, predictions, class_names, output_folder)


def create_artifact_zip(output_folder):
    """Create a ZIP next to a completed evidence directory and return its path."""
    folder = Path(output_folder)
    archive = shutil.make_archive(str(folder), "zip", root_dir=folder.parent, base_dir=folder.name)
    return Path(archive)

def main():
    """CLI-compatible full run; the notebook exposes the same calls stage by stage."""
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
        selected_result = validate_majority(probabilities, validation_loader)
        selected_result["probabilities"] = probabilities.numpy()
        best_epoch = None
        extra_summary = {"majority_class": majority_class}
    else:
        train_neural_mode(mode, train_loader, validation_loader, torch.device("cuda"), output_folder)
        _, best_epoch, selected_result = evaluate_best_checkpoint(mode, validation_loader, torch.device("cuda"), output_folder)
        extra_summary = {}

    metrics = validation_metrics(selected_result["labels"], selected_result["predictions"], selected_result.get("probabilities"))
    summary = {"mode": mode, "best_epoch": best_epoch, "best_validation_macro_f1": metrics["macro_f1"], "runtime_seconds": time.time() - started, "test_evaluated": False, **extra_summary}
    save_json(output_folder / "metrics.json", metrics)
    save_json(output_folder / "summary.json", summary)
    class_names = class_names_from_audit(CONFIG["audit_manifest_path"])
    save_validation_diagnostics(selected_result["labels"], selected_result["predictions"], metrics, validation_loader, class_names, output_folder)
    archive = create_artifact_zip(output_folder)
    print("Saved validation-only evidence to", output_folder)
    print("Saved artifact archive to", archive)
    print(summary)


if __name__ == "__main__":
    main()
