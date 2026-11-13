"""Validation-only metric and provenance serialization for multiclass baselines."""
from __future__ import annotations

from typing import Any
import numpy as np
from sklearn.metrics import classification_report, f1_score, roc_auc_score

from .data import CLASS_NAMES, class_count_rows


def metric_payload(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    """Calculate required validation metrics with class prevalence.

    ROC-AUC is macro one-vs-rest using probability scores. The value is ``None``
    when the validation labels lack any class, rather than producing a misleading
    scalar. Precision/recall use zero_division=0 so an unpredicted class is
    explicitly reported as zero.
    """
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.shape != (len(y_true), len(CLASS_NAMES)):
        raise ValueError("Expected one probability row and eight scores per validation example.")
    y_pred = probabilities.argmax(axis=1)
    report = classification_report(
        y_true, y_pred, labels=list(CLASS_NAMES), target_names=[CLASS_NAMES[i] for i in CLASS_NAMES],
        output_dict=True, zero_division=0,
    )
    try:
        auc: float | None = float(roc_auc_score(y_true, probabilities, labels=list(CLASS_NAMES), multi_class="ovr", average="macro"))
    except ValueError:
        auc = None
    per_class = []
    prevalence_by_index = {row["class_index"]: row for row in class_count_rows(y_true)}
    for index, name in CLASS_NAMES.items():
        row = report[name]
        prevalence = prevalence_by_index[index]
        per_class.append({
            "class_index": index, "class_name": name, "support": int(row["support"]),
            "prevalence": prevalence["prevalence"], "precision": float(row["precision"]),
            "recall": float(row["recall"]), "f1": float(row["f1-score"]),
        })
    return {
        "n_examples": int(len(y_true)),
        "accuracy": float((y_pred == y_true).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(CLASS_NAMES), average="macro", zero_division=0)),
        "macro_ovr_roc_auc": auc,
        "per_class": per_class,
    }
