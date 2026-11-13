"""Validation-only classification metrics and serializable reports."""
from __future__ import annotations
import numpy as np

CLASS_NAMES = ["basophil", "eosinophil", "erythroblast", "immature_granulocytes", "lymphocyte", "monocyte", "neutrophil", "platelet"]

def classification_report(y_true, logits) -> dict:
    """Calculate accuracy, per-class precision/recall/F1, macro-F1 and multiclass OVR AUC."""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score, roc_auc_score
    import torch
    y_true = np.asarray(y_true, dtype=int)
    scores = torch.softmax(torch.as_tensor(logits), dim=1).numpy()
    pred = scores.argmax(axis=1)
    p, r, f, support = precision_recall_fscore_support(y_true, pred, labels=range(8), zero_division=0)
    result = {"accuracy": float(accuracy_score(y_true, pred)), "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0),),
              "per_class": {CLASS_NAMES[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(support[i])} for i in range(8)}}
    try: result["roc_auc_ovr_macro"] = float(roc_auc_score(y_true, scores, labels=list(range(8)), multi_class="ovr", average="macro"))
    except ValueError: result["roc_auc_ovr_macro"] = None
    return result
