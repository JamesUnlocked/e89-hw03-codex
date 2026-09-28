# HW03 Problem 4: Dialog Summary

## Project instructions

You asked to build the â€œBuilding an Image Classifier with PyTorchâ€ example from
chapter 10 of GÃ©ron's *Hands-On Machine Learning with Scikit-Learn and PyTorch*
(2025), one script at a time. Scripts were to use `stepN_<topic>.py` names,
import earlier steps when needed, and put executable entry-point code under
`if __name__ == "__main__":`. Each step was to be run, corrected if necessary,
and committed with a `Step N: <topic>` message.

## Requests, deliverables, and commits

| Request | Generated or updated files and behavior | Commit |
| --- | --- | --- |
| **Step 1: Data.** Load FashionMNIST with the specified torchvision v2 transforms; seed with 42; split 60,000 images into 55,000 training and 5,000 validation samples; provide loaders, class names, and device selection; print a sample. | Created `step1_data.py` with `get_loaders(batch_size=32)` and `get_device()`. Training is shuffled; validation and test are not. Transforms convert images to scaled float32 tensors. Added `.gitignore` for `datasets/` and `__pycache__/`. | `e2dc1f5` â€” `Step 1: data` |
| **Step 2: Model.** Define `ImageClassifier(nn.Module)` with Flatten, Linear(784, 300), ReLU, Linear(300, 100), ReLU, and Linear(100, 10); return raw logits; print parameter count and batch output shape. | Created `step2_model.py`, importing data and device helpers from Step 1. The main block seeds PyTorch with 42 and runs one batch through the model on the selected device. | `1c9eefa` â€” `Step 2: model` |
| **Step 3: Training.** Implement `train(model, optimizer, loss_fn, metric, train_loader, valid_loader, n_epochs)`; report mean loss and training/validation accuracy each epoch; return history; train 20 epochs with SGD at 0.1 and CrossEntropyLoss; save history and weights. | Created `step3_train.py` and generated `history.json` and `model.pt`. The loop uses sample-weighted mean loss and torchmetrics multiclass accuracy, resetting the metric between training and validation. Validation uses evaluation mode and disables gradients. Added `model.pt` to `.gitignore`; committed the script, history, and ignore update, leaving weights local. | `e630ee1` â€” `Step 3: train` |
| **Step 4: Plot.** Load history, plot training and validation accuracy together with title, labels, legend, and grid; save the PNG and call `plt.show()`. | Created `step4_plot_accuracy.py` and generated `training_accuracy.png`. Inspected the saved plot and committed both files. | `9ac7e8d` â€” `Step 4: plot accuracy` |
| **Step 5: Evaluation.** Rebuild the model, load weights, report test accuracy, and inspect the first three validation images with predicted indices/names, true labels, rounded softmax probabilities, and top-four classes. | Created and ran `step5_evaluate.py`. It loads the state dictionary with `weights_only=True`, evaluates the test set, and prints all requested prediction details. | `dc66dc8` â€” `Step 5: evaluate` |
| **Dependencies and documentation.** Add requirements and a README listing scripts in execution order, with one-line descriptions and run commands; commit and push everything to `origin main`. | Created `requirements.txt` and `README.md`. Requirements record torch 2.14.0, torchvision 0.29.0, torchmetrics 1.9.0, and matplotlib 3.10.9. The README documents setup, execution order, outputs, ignored files, and results. Pushed all five step commits and this documentation commit to `origin main`. | `5692cdc` â€” `Add requirements and execution guide` |
| **Dialog summary.** Record every request, files and commit hashes, issues and resolutions, and final metrics; save, commit, and push the summary. | Created this file, `HW03_Prob4_dialog_summary.md`, for a separate commit and push. Its commit hash is reported in the assistant's completion message because a commit cannot contain its own final hash. | Commit message: `Add HW03 Problem 4 dialog summary` |

## Execution and verification

- **Step 1:** FashionMNIST downloaded successfully. The first training sample had shape `torch.Size([1, 28, 28])`, dtype `torch.float32`, and class `Ankle boot`.
- **Step 2:** The model had **266,610 parameters** and returned `torch.Size([32, 10])` for a batch of 32 images, with no softmax in the model.
- **Step 3:** All 20 epochs completed on CPU. Verified that all three history lists contained 20 finite values, that the weights reloaded successfully, and that the reloaded model produced the expected output shape. Confirmed Git ignored the dataset and weights.
- **Step 4:** The PNG showed both accuracy curves across epochs 1â€“20 with the requested formatting. The script reached the interactive `plt.show()` call.
- **Step 5:** Evaluation completed successfully. Predicted validation indices were `[7, 4, 2]`, corresponding to `Sneaker`, `Coat`, and `Pullover`; the true labels were also `[7, 4, 2]`.
- **Repository:** After the documentation push, local `main` and `origin/main` both pointed to `5692cdc`, and the working tree was clean.

## Errors, issues, and resolutions

- **No Python runtime errors or missing dependencies occurred** in the five scripts. No code fixes were required after running them.
- **Initial download and CPU training took time.** The dataset download was allowed to finish, and training was monitored through all 20 epochs without shortening the requested run.
- **Git emitted LF-to-CRLF warnings on Windows.** These were line-ending notices, not failures; commits succeeded and whitespace checks passed.
- **A review command, `git diff --no-index`, returned exit code 1.** This is the expected result when the compared files differ; it displayed the newly created Step 1 script and did not indicate a script error.
- **Displaying the plot required permission to launch a GUI outside the sandbox.** Approval was requested through the execution tool, and the plotting command ran. `plt.show()` kept the process open while the plot window was displayed, as expected; the PNG had already been saved and was committed. At the Step 4 completion report, the window remained open.
- **Large generated assets were excluded intentionally.** `datasets/`, `model.pt`, and `__pycache__/` remain ignored. `history.json` and `training_accuracy.png` were committed. Reproducing evaluation from a fresh checkout requires running Step 3 to create `model.pt`.
- **The push to `origin main` succeeded** without merge conflicts or reported authentication/network errors.

## Final results

| Quantity | Result |
| --- | --- |
| Training / validation / test samples | 55,000 / 5,000 / 10,000 |
| Model parameters | **266,610** |
| Training epochs | **20** |
| Final mean training loss | **0.1888** |
| Final training accuracy | **92.80%** |
| Final validation accuracy | **88.38%** |
| Best validation accuracy | **88.84%**, at epoch 16 |
| Test accuracy of final saved weights | **88.61%** |
| First three validation predictions correct | **3 / 3** |

The saved weights are from the final epoch, rather than a best-validation
checkpoint. The repository remote is
`https://github.com/JamesUnlocked/e89-hw03-codex.git`, on branch `main`.

