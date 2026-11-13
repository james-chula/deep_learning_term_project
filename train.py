"""Plain training and validation loops for one epoch at a time."""
import torch
from sklearn.metrics import f1_score


def train_one_epoch(model, loader, optimizer, loss_function, device):
    """Update model weights once using every batch in the training loader."""
    model.train()
    total_loss = 0.0
    correct = 0
    examples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.view(-1).long().to(device)

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
    """Measure validation loss, accuracy, and macro-F1 without updating weights."""
    model.eval()
    total_loss = 0.0
    correct = 0
    examples = 0
    true_labels = []
    predicted_labels = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.view(-1).long().to(device)
        scores = model(images)
        loss = loss_function(scores, labels)
        predictions = scores.argmax(dim=1)

        examples += labels.size(0)
        total_loss += loss.item() * labels.size(0)
        correct += (predictions == labels).sum().item()
        true_labels.extend(labels.cpu().tolist())
        predicted_labels.extend(predictions.cpu().tolist())

    return {
        "loss": total_loss / examples,
        "accuracy": correct / examples,
        "macro_f1": f1_score(true_labels, predicted_labels, average="macro"),
    }
