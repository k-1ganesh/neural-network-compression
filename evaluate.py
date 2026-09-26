import torch
import torch.nn as nn


def evaluate_model(model, data_loader, device):

    model.eval()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in data_loader:

            images = images.to(device,non_blocking=True)

            labels = labels.to(device,non_blocking=True)

            logits = model(images)

            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)

            predictions = logits.argmax(dim=1)

            correct += (predictions == labels).sum().item()

            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
    }


def count_parameters(model): # Count trainable parameters
    return sum(p.numel() for p in model.parameters() if p.requires_grad)