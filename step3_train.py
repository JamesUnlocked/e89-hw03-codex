"""Train the FashionMNIST classifier and save its history and weights."""

import json

import torch
from torch import nn
from torchmetrics import Accuracy

from step1_data import get_device, get_loaders
from step2_model import ImageClassifier


def train(model, optimizer, loss_fn, metric, train_loader, valid_loader, n_epochs):
    """Train with a mean-reduced loss and record sample-weighted epoch metrics."""
    device = next(model.parameters()).device
    metric = metric.to(device)
    history = {"train_loss": [], "train_acc": [], "valid_acc": []}

    for epoch in range(1, n_epochs + 1):
        model.train()
        metric.reset()
        total_loss = 0.0
        total_samples = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)
            metric.update(logits.detach(), labels)

        train_loss = total_loss / total_samples
        train_acc = metric.compute().item()

        model.eval()
        metric.reset()
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                metric.update(model(images), labels)
        valid_acc = metric.compute().item()
        metric.reset()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["valid_acc"].append(valid_acc)
        print(
            f"Epoch {epoch}/{n_epochs}, train loss: {train_loss:.4f}, "
            f"train metric: {train_acc:.4f}, valid metric: {valid_acc:.4f}",
            flush=True,
        )

    return history


if __name__ == "__main__":
    torch.manual_seed(42)
    device = get_device()
    model = ImageClassifier().to(device)
    train_loader, valid_loader, test_loader, class_names = get_loaders()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.CrossEntropyLoss()
    metric = Accuracy(task="multiclass", num_classes=10)

    history = train(
        model, optimizer, loss_fn, metric, train_loader, valid_loader, n_epochs=20
    )
    with open("history.json", "w", encoding="utf-8") as history_file:
        json.dump(history, history_file, indent=2)
        history_file.write("\n")
    torch.save(model.state_dict(), "model.pt")
