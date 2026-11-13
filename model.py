"""Create the pretrained eight-class ResNet-18 used by neural modes."""

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def create_model(num_classes=8, mode="finetune"):
    """Return ImageNet-pretrained ResNet-18 with an eight-class output layer.

    ``linear_probe`` freezes every backbone parameter; ``finetune`` leaves all
    parameters trainable. The majority baseline does not create a model.
    """
    if mode not in {"linear_probe", "finetune"}:
        raise ValueError("Model creation is only valid for linear_probe or finetune.")
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    if mode == "linear_probe":
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.fc.parameters():
            parameter.requires_grad_(True)
    return model


def majority_probabilities(train_labels, validation_count, num_classes=8):
    """Predict the majority training class for every validation example."""
    labels = torch.as_tensor(train_labels).view(-1).long()
    majority = int(torch.bincount(labels, minlength=num_classes).argmax().item())
    probabilities = torch.zeros((validation_count, num_classes), dtype=torch.float32)
    probabilities[:, majority] = 1.0
    return probabilities, majority
