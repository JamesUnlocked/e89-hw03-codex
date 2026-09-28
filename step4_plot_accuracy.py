"""Plot training and validation accuracy from the saved training history."""

import json

import matplotlib.pyplot as plt


if __name__ == "__main__":
    with open("history.json", encoding="utf-8") as history_file:
        history = json.load(history_file)

    epochs = range(1, len(history["train_acc"]) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_acc"], marker="o", label="Training accuracy")
    ax.plot(epochs, history["valid_acc"], marker="s", label="Validation accuracy")
    ax.set_title("FashionMNIST Training and Validation Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(list(epochs))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("training_accuracy.png", dpi=150)
    plt.show()
