"""Evaluate saved FashionMNIST weights and inspect validation predictions."""

import torch
from torchmetrics import Accuracy

from step1_data import get_device, get_loaders
from step2_model import ImageClassifier


if __name__ == "__main__":
    device = get_device()
    model = ImageClassifier().to(device)
    model.load_state_dict(
        torch.load("model.pt", map_location=device, weights_only=True)
    )
    model.eval()
    train_loader, valid_loader, test_loader, class_names = get_loaders()
    metric = Accuracy(task="multiclass", num_classes=10).to(device)

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            metric.update(model(images), labels)
        print(f"Test accuracy: {metric.compute().item():.4%}")

        samples = [valid_loader.dataset[i] for i in range(3)]
        images = torch.stack([image for image, label in samples]).to(device)
        true_labels = [label for image, label in samples]
        probabilities = model(images).softmax(dim=1).cpu()
        predictions = probabilities.argmax(dim=1).tolist()
        top_probabilities, top_indices = probabilities.topk(4, dim=1)

    print(f"Predicted class indices: {predictions}")
    print(f"Predicted class names: {[class_names[i] for i in predictions]}")
    print(f"True labels: {true_labels}")
    print(f"True class names: {[class_names[i] for i in true_labels]}")
    print(f"Probability columns: {class_names}")
    print("Softmax probabilities (rounded to 3 decimals):")
    for row in probabilities.tolist():
        print("[" + ", ".join(f"{value:.3f}" for value in row) + "]")

    print("Top-4 classes per validation image:")
    for sample_index, (indices, values) in enumerate(
        zip(top_indices.tolist(), top_probabilities.tolist()), start=1
    ):
        print(f"Image {sample_index}:")
        for class_index, probability in zip(indices, values):
            print(f"  {class_index}: {class_names[class_index]} ({probability:.3f})")
