# FashionMNIST Image Classifier with PyTorch

A step-by-step implementation of the image classifier example from chapter 10 of
Aurélien Géron's *Hands-On Machine Learning with Scikit-Learn and PyTorch* (2025).

## Setup

Install the dependencies from the project directory:

```sh
python -m pip install -r requirements.txt
```

The requirements record the package versions used for this project. The scripts
automatically select CUDA, MPS, or CPU, depending on availability.

## Execution order

Run these commands from the project directory:

1. `python step1_data.py` — Download FashionMNIST into `datasets/`, split training data into 55,000 training and 5,000 validation images, and print a sample's shape, dtype, and class.
2. `python step2_model.py` — Build the 784 → 300 → 100 → 10 classifier and print its parameter count and one batch's output shape.
3. `python step3_train.py` — Train for 20 epochs with SGD, print loss and accuracy, and save `history.json` and `model.pt`.
4. `python step4_plot_accuracy.py` — Plot training and validation accuracy from `history.json`, save `training_accuracy.png`, and display the plot (close the window to exit).
5. `python step5_evaluate.py` — Load `model.pt`, report test accuracy, and print predictions, true labels, softmax probabilities, and top-four classes for the first three validation images.

Step 3 must run before Step 5 to generate the weights. Step 4 uses the saved
history, which is also included in this repository. Downloads require internet
access on the first run; subsequent runs reuse the dataset.

`datasets/`, `model.pt`, and Python cache files are ignored by Git. Training history
and the accuracy plot are committed. The recorded run achieved 88.61% test accuracy;
results may vary across devices and package builds.
