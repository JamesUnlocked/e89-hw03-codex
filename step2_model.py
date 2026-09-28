"""Define a fully connected FashionMNIST classifier that returns raw logits."""

import torch
from torch import nn

from step1_data import get_device, get_loaders


class ImageClassifier(nn.Module):
    """Map 28-by-28 grayscale images to logits for ten classes."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 300),
            nn.ReLU(),
            nn.Linear(300, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
        )

    def forward(self, images):
        return self.layers(images)


if __name__ == "__main__":
    torch.manual_seed(42)
    device = get_device()
    model = ImageClassifier().to(device)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_loader, valid_loader, test_loader, class_names = get_loaders()
    images, labels = next(iter(train_loader))
    with torch.no_grad():
        logits = model(images.to(device))
    print(f"Output shape: {logits.shape}")
