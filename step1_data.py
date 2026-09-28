"""Load FashionMNIST and select the available PyTorch device."""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import FashionMNIST
from torchvision.transforms import v2 as T


def get_loaders(batch_size=32):
    """Return training, validation, and test loaders, plus class names."""
    transform = T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
    ])
    full_train_data = FashionMNIST(
        root="datasets", train=True, download=True, transform=transform
    )
    test_data = FashionMNIST(
        root="datasets", train=False, download=True, transform=transform
    )

    torch.manual_seed(42)
    train_data, valid_data = random_split(full_train_data, [55_000, 5_000])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    return train_loader, valid_loader, test_loader, full_train_data.classes


def get_device():
    """Return the preferred available device: CUDA, MPS, or CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


if __name__ == "__main__":
    train_loader, valid_loader, test_loader, class_names = get_loaders()
    image, label = train_loader.dataset[0]
    print(f"Shape: {image.shape}")
    print(f"Dtype: {image.dtype}")
    print(f"Class: {class_names[label]}")
