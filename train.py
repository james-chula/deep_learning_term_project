"""Direct train and validation loops. Validation never updates model weights."""

import torch
from sklearn.metrics import f1_score


def train_one_epoch(model, loader, optimizer, loss_function, device):
    """Update model weights using only training batches."""
    model.train()
    total_loss = 0.0
    correct = 0
    examples = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.view(-1).long().to(device, non_blocking=True)
        optimizer.zero_grad()
        scores = model(images)
        loss = loss_function(scores, labels)
        loss.backward()
        optimizer.step()
        examples += labels.size(0)
        total_loss += loss.item() * labels.size(0)
        correct += (scores.argmax(dim=1) == labels).sum().item()
    return {"loss": total_loss / examples, "accuracy": correct / examples}


@torch.no_grad()
def validate(model, loader, loss_function, device):
    """Measure validation metrics without gradients or optimizer updates."""
    model.eval()
    total_loss = 0.0
    correct = 0
    examples = 0
    true_labels, predicted_labels, probabilities = [], [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.view(-1).long().to(device, non_blocking=True)
        scores = model(images)
        loss = loss_function(scores, labels)
        predictions = scores.argmax(dim=1)
        examples += labels.size(0)
        total_loss += loss.item() * labels.size(0)
        correct += (predictions == labels).sum().item()
        true_labels.extend(labels.cpu().tolist())
        predicted_labels.extend(predictions.cpu().tolist())
        probabilities.extend(torch.softmax(scores, dim=1).cpu().tolist())
    return {
        "loss": total_loss / examples,
        "accuracy": correct / examples,
        "macro_f1": f1_score(true_labels, predicted_labels, average="macro", zero_division=0),
        "labels": true_labels,
        "predictions": predicted_labels,
        "probabilities": probabilities,
    }


def validate_majority(probabilities, validation_loader):
    """Evaluate fixed majority predictions against validation labels only."""
    labels = []
    for _, batch_labels in validation_loader:
        labels.extend(batch_labels.view(-1).tolist())
    predictions = probabilities.argmax(dim=1).tolist()
    return {
        "accuracy": float(sum(a == b for a, b in zip(labels, predictions)) / len(labels)),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "labels": labels,
        "predictions": predictions,
    }
