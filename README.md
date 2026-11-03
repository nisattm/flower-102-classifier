# Flower 102 Classification — TorchScript CPU Submission

This project classifies a photograph into one of **102 flower categories**. We trained and compared pretrained image classifiers on the flower dataset supplied for the course, then selected EfficientNet-B0 for CPU inference. The instructor's updated submission contract requires **exactly one file named `model.pt`**. Our deliverable is [`submission_ready/model.pt`](submission_ready/model.pt); it contains the network, trained weights, input normalization, and softmax in one TorchScript file. No dataset, Python source file, or separate weight file is needed by the evaluator.

## Project team

| Student | Student number |
|---|---:|
| Nisa Atım | 220205014 |
| Ceren Yüksel | 220201039 |
| Mustafa Eymen Aslan | 210201004 |
| Efekan Coşkun | 220201024 |

## Official submission file and interface

**Submit only [`submission_ready/model.pt`](submission_ready/model.pt).** The `submission_ready/` directory contains no other file. The earlier `submission/model.pth` package is retained only to reproduce the original experiments; it is **not accepted** by the updated evaluator.

| Contract item | Implemented behavior |
|---|---|
| File | TorchScript created with `torch.jit.trace`; load with `torch.jit.load("model.pt", map_location="cpu")` |
| Input | `float32` RGB tensor of shape `[B, 3, 224, 224]` with values in `[0, 1]` |
| Image processing performed by evaluator | Resize shorter side to 256, center-crop to 224, convert to a 0–1 tensor |
| Processing inside the file | Apply the training mean/std normalization `(x - 0.5) / 0.5`, run EfficientNet-B0, apply softmax |
| Output | One `[B, 102]` tensor of probabilities in `[0, 1]`; each row sums to 1 |
| Class order | Output column `i` represents class `i + 1`, for classes 1 through 102 |

The file was loaded and run on CPU with batch sizes 1, 2, and 8. The saved graph matched the eager model on random 0–1 inputs, and a separate validation run using the documented resize/crop with assumed bilinear interpolation checked the numeric class order on real flower images. Export and validation commands are described below.

## Evaluation objective

The competition combines hidden-test top-1 accuracy (60%), CPU inference speed (25%), and saved model size (15%). The guideline describes an Intel Core i7 computer with 16 GB RAM and no GPU. Its current **proposed**, unconfirmed reference values are 50 ms per image and 35 MB. We therefore selected a model using validation accuracy, measured CPU latency, and the actual inference file size. The organizer's hidden test and final timing procedure were not available to us.

## Data and leakage control

The supplied archive contains **8,189 images** across 102 classes: 6,552 in `train`, 818 in `valid`, and 819 in `test`. We used `train + valid` for model development and reserved the 819-image `test` split for one final accuracy evaluation after the model and training recipe were fixed.

Before making validation folds, we checked labels and class names against the Oxford source, reviewed exact and visually similar images, and removed confirmed duplicates, conflicting labels, and development images closely overlapping test images. The cross-split image comparison served only to prevent leakage; test accuracy was not used to select a model. This left **7,339 development images**. The frozen three-fold split has validation sizes of 2,478, 2,447, and 2,414. The final model was trained on all 7,339 development images; the 819 test images were never used for training. Audit details are in [`artifacts/data/audit.json`](artifacts/data/audit.json) and the full [experiment report](EXPERIMENT_REPORT.md).

## Models and selection

We initially explored ConvNeXtV2 Tiny as a high-accuracy teacher, EfficientNet-B0, and MobileNetV4 Conv Small. We also tested a higher backbone learning rate and teacher-to-MobileNet distillation. Distillation improved fold 0 by only two correct images relative to the tuned direct MobileNet run, so that single-fold result was not enough to select it. ConvNeXtV2 Tiny achieved strong fold-0 accuracy but exceeded the proposed model-size and local CPU-speed references.

The two complete three-fold finalists were compared **before the one-file TorchScript format was announced**:

| Candidate | Correct across 7,339 validation images | Top-1 | Earlier FP16 weight file | Earlier local CPU mean, disk included |
|---|---:|---:|---:|---:|
| **EfficientNet-B0** | **7,226** | **98.4603%** | 8.477 MB | 30.210 ms/image |
| MobileNetV4 Conv Small | 7,184 | 97.8880% | 5.391 MB | 18.073 ms/image |

These are measurements of the selected fold checkpoints on our laptop, with 50 warm-up predictions and five rounds of 1,000 images. They are not measurements of the new TorchScript file or the organizer's computer. Under the guideline's proposed score thresholds, both models reached the saturated speed and size scores locally; EfficientNet-B0 was selected for its higher three-fold accuracy. The selection and frozen recipe were recorded [before the internal test was opened](FINAL_MODEL_DECISION.md).

The final EfficientNet-B0 used the `efficientnet_b0.ra4_e3600_r224_in1k` pretrained variant, 224 × 224 input, seed 42, AdamW, weight decay 0.05, label smoothing 0.1, three head-only epochs, backbone learning rate `1e-4`, head learning rate `1e-3`, and cosine decay. It trained for **14 fixed epochs** on all 7,339 development images with batch size 8 and effective batch size 32. Training checkpoints were saved after each epoch so an interrupted run could resume.

## Measured results and their scope

| Measurement | Result |
|---|---:|
| One-time evaluation of the selected model on the supplied 819-image internal test, using the earlier preprocessing | **809 / 819 = 98.7790% top-1** |
| New-format compatibility check on the frozen fold-0 checkpoint and validation images | **2,454 / 2,478 = 99.0315% top-1** |
| Official TorchScript `submission_ready/model.pt` file | **17.242 MB** |
| Official TorchScript file on our CPU, model forward only, batch size 1 | **23.106 ms/image mean** |

The new TorchScript speed check used 50 warm-ups and 5 × 1,000 timed forward passes with six CPU threads; it **excludes image reading and evaluator preprocessing**. Results are in [`artifacts/benchmarks/efficient_final_torchscript_cpu.json`](artifacts/benchmarks/efficient_final_torchscript_cpu.json). The earlier 8.471 MB weight package and its 43.196 ms/image disk-included timing are recorded in [`EXPERIMENT_REPORT.md`](EXPERIMENT_REPORT.md), but refer to a different file and timing scope. The fold-0 compatibility check uses a separate validation checkpoint, not the final model; it followed the documented 256/224 geometry with bilinear interpolation because the PDF does not specify interpolation. Its result is [recorded here](artifacts/benchmarks/efficient_f0_torchscript_evaluator_fold0.json). **No competition hidden-test result is claimed.** The internal 819-image test was not reopened to tune the new format.

## Run and verify the new model

The evaluator needs **only the `model.pt` file** and PyTorch. Its required call is:

```python
import torch

model = torch.jit.load("model.pt", map_location="cpu").eval()
probabilities = model(torch.rand(2, 3, 224, 224))
print(probabilities.shape, probabilities.sum(dim=1))
# torch.Size([2, 102]); each row sum is approximately 1
```

The input must already be RGB, resized to a 256-pixel short side, center-cropped to 224 × 224, and converted to float values in `[0, 1]`. **Do not apply mean/std normalization outside the model**: it is embedded in the TorchScript graph. Output column 0 is class 1, and output column 101 is class 102.

To try a local photo, use the optional helper script in this repository. The helper is **not** part of the one-file submission. On Windows PowerShell, from the repository root:

```powershell
git clone https://github.com/nisattm/flower-102-classifier.git
cd flower-102-classifier
py -3.12 -m venv .venv
./.venv/Scripts/python.exe -m pip install -r submission/requirements.txt
./.venv/Scripts/python.exe predict_torchscript.py --image "C:/Photos/flower.jpg"
```

This prints the numeric 1–102 `class_id`, `class_name`, and `confidence`. Package installation needs internet access; the model's inference does not. The earlier [`submission/` package](submission/) remains available for reproducing previous experiments but **must not be uploaded in place of `model.pt`**.

## Repository contents and reproducibility

| Location | Purpose |
|---|---|
| [`submission_ready/model.pt`](submission_ready/model.pt) | **Only file to submit** to the instructor's evaluator |
| `export_torchscript_submission.py` | Reproducible conversion from the frozen checkpoint to the one-file TorchScript format |
| `predict_torchscript.py`, `evaluate_torchscript_format.py` | Local image demo and development-fold compatibility check; not submitted |
| [`submission/`](submission/) | Earlier Python inference package, retained for reproducibility; not accepted by the new evaluator |
| [`EXPERIMENT_REPORT.md`](EXPERIMENT_REPORT.md) | Recorded training, validation, export, and benchmark results |
| [`FINAL_MODEL_DECISION.md`](FINAL_MODEL_DECISION.md) | Model-selection evidence and frozen training recipe |
| [`CPU_TARGET.md`](CPU_TARGET.md) | Interpretation of the competition's CPU scoring guideline |
| [`artifacts/data/folds.jsonl`](artifacts/data/folds.jsonl) | Frozen validation assignments |
| `train.py`, `train_final.py`, `evaluate_checkpoint.py` | Training and evaluation scripts |

The original image archive, pretrained-weight cache, and intermediate checkpoints are intentionally not in Git. Reproducing training requires the instructor-supplied `flower_data` archive; the evaluator only needs `model.pt`. The final training command was:

```powershell
./.venv/Scripts/python.exe train_final.py --data "C:/path/to/flower_data" --run artifacts/runs/efficient_final_all_dev --model efficient --batch 8 --workers 2 --epochs 14 --backbone-lr 0.0001
```

The new one-file artifact was exported from the frozen final weights without retraining:

```powershell
./.venv/Scripts/python.exe export_torchscript_submission.py --source submission/model.pth --output submission_ready/model.pt
```

`evaluate_torchscript_format.py` checks the new interface on frozen development folds without reading the supplied internal test. `benchmark_torchscript_submission.py` records forward latency separately from preprocessing and file I/O.

For later model versions, compare changes on the frozen development folds or genuinely new validation data. The supplied 819-image internal test was evaluated once after selection and should not be reused for repeated tuning.
