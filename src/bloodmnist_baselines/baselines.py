"""Mandatory majority-class and frozen-pretrained-ResNet-18 linear-probe baselines."""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

from .data import CLASS_NAMES


@dataclass(frozen=True)
class LinearProbeConfig:
    """Configuration for fixed-feature multinomial logistic regression."""

    batch_size: int = 128
    num_workers: int = 2
    seed: int = 2026
    logistic_c: float = 1.0
    max_iter: int = 1000
    pretrained_weights: str = "ResNet18_Weights.IMAGENET1K_V1"


def _loader(dataset: Dataset, config: LinearProbeConfig, *, shuffle: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle, num_workers=config.num_workers,
                      pin_memory=True, generator=generator)


def _extract_features(dataset: Dataset, encoder: nn.Module, device: torch.device, config: LinearProbeConfig) -> tuple[np.ndarray, np.ndarray]:
    """Extract one immutable ImageNet feature vector per sample; encoder stays eval/frozen."""
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.inference_mode():
        for images, target in _loader(dataset, config, shuffle=False):
            output = encoder(images.to(device, non_blocking=True)).cpu().numpy()
            features.append(output)
            labels.append(np.asarray(target).reshape(-1))
    return np.concatenate(features), np.concatenate(labels).astype(np.int64)


def majority_class_baseline(train_labels: np.ndarray, val_labels: np.ndarray) -> tuple[dict[str, Any], dict[str, Any]]:
    """Predict the most prevalent *training* class for every validation image."""
    train_labels = np.asarray(train_labels, dtype=np.int64).reshape(-1)
    majority = int(np.bincount(train_labels, minlength=len(CLASS_NAMES)).argmax())
    probabilities = np.zeros((len(val_labels), len(CLASS_NAMES)), dtype=np.float64)
    probabilities[:, majority] = 1.0
    return probabilities, {"majority_class_index": majority, "majority_class_name": CLASS_NAMES[majority]}


def frozen_resnet18_linear_probe(train_dataset: Dataset, val_dataset: Dataset, device: torch.device, config: LinearProbeConfig) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit only multinomial logistic-regression parameters on frozen ResNet-18 features.

    Pretrained ImageNet weights are downloaded by torchvision in Colab if absent.
    No encoder gradients or encoder optimizer are constructed.
    """
    start = perf_counter()
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    encoder = nn.Sequential(*list(model.children())[:-1], nn.Flatten()).to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    train_x, train_y = _extract_features(train_dataset, encoder, device, config)
    val_x, val_y = _extract_features(val_dataset, encoder, device, config)
    probe = LogisticRegression(C=config.logistic_c, max_iter=config.max_iter, random_state=config.seed,
                               multi_class="multinomial", solver="lbfgs")
    probe.fit(train_x, train_y)
    probabilities = probe.predict_proba(val_x)
    if not np.array_equal(probe.classes_, np.arange(len(CLASS_NAMES))):
        raise ValueError("Training data must include every BloodMNIST class for the fixed probability schema.")
    return probabilities, {
        "encoder": "torchvision.models.resnet18", "encoder_weights": config.pretrained_weights,
        "encoder_frozen": True, "feature_dimension": int(train_x.shape[1]),
        "probe": "sklearn.linear_model.LogisticRegression", "solver": "lbfgs", "multi_class": "multinomial",
        "C": config.logistic_c, "max_iter": config.max_iter, "runtime_seconds": perf_counter() - start,
        "validation_label_checksum": int(val_y.sum()),
    }
