# Flower 102 Classification — CPU Submission

This project classifies a photograph into one of **102 flower categories**. We trained and compared several pretrained image classifiers on the flower dataset supplied for the course, then selected an EfficientNet-B0 model for CPU inference. The repository includes the final model, inference code, fixed class mapping, experiment records, and instructions to reproduce a prediction. **No training images are required to run the submitted model.**

## Project team

| Student | Student number |
|---|---:|
| Nisa Atım | 220205014 |
| Ceren Yüksel | 220201039 |
| Mustafa Eymen Aslan | 210201004 |
| Efekan Coşkun | 220201024 |

## Evaluation objective

The competition combines hidden-test top-1 accuracy (60%), CPU inference speed (25%), and saved model size (15%). The guideline describes an Intel Core i7 computer with 16 GB RAM and no GPU. Its current **proposed**, unconfirmed reference values are 50 ms per image and 35 MB. We therefore selected a model using validation accuracy, measured CPU latency, and the actual inference file size. The organizer's hidden test and final timing procedure were not available to us.

## Data and leakage control

The supplied archive contains **8,189 images** across 102 classes: 6,552 in `train`, 818 in `valid`, and 819 in `test`. We used `train + valid` for model development and reserved the 819-image `test` split for one final accuracy evaluation after the model and training recipe were fixed.

Before making validation folds, we checked labels and class names against the Oxford source, reviewed exact and visually similar images, and removed confirmed duplicates, conflicting labels, and development images closely overlapping test images. The cross-split image comparison served only to prevent leakage; test accuracy was not used to select a model. This left **7,339 development images**. The frozen three-fold split has validation sizes of 2,478, 2,447, and 2,414. The final model was trained on all 7,339 development images; the 819 test images were never used for training. Audit details are in [`artifacts/data/audit.json`](artifacts/data/audit.json) and the full [experiment report](EXPERIMENT_REPORT.md).

## Models and selection

We initially explored ConvNeXtV2 Tiny as a high-accuracy teacher, EfficientNet-B0, and MobileNetV4 Conv Small. We also tested a higher backbone learning rate and teacher-to-MobileNet distillation. Distillation improved fold 0 by only two correct images relative to the tuned direct MobileNet run, so that single-fold result was not enough to select it. ConvNeXtV2 Tiny achieved strong fold-0 accuracy but exceeded the proposed model-size and local CPU-speed references.

The two complete three-fold finalists were:

| Candidate | Correct across 7,339 validation images | Top-1 | FP16 model file | Local CPU mean, disk included |
|---|---:|---:|---:|---:|
| **EfficientNet-B0** | **7,226** | **98.4603%** | 8.477 MB | 30.210 ms/image |
| MobileNetV4 Conv Small | 7,184 | 97.8880% | 5.391 MB | 18.073 ms/image |

These are measurements of the selected fold checkpoints on our laptop, with 50 warm-up predictions and five rounds of 1,000 images. They are not measurements on the organizer's computer. Under the guideline's proposed score thresholds, both models reached the saturated speed and size scores locally; EfficientNet-B0 was selected for its higher three-fold accuracy. The selection and frozen recipe were recorded [before the internal test was opened](FINAL_MODEL_DECISION.md).

The final EfficientNet-B0 used the `efficientnet_b0.ra4_e3600_r224_in1k` pretrained variant, 224 × 224 input, seed 42, AdamW, weight decay 0.05, label smoothing 0.1, three head-only epochs, backbone learning rate `1e-4`, head learning rate `1e-3`, and cosine decay. It trained for **14 fixed epochs** on all 7,339 development images with batch size 8 and effective batch size 32. Training checkpoints were saved after each epoch so an interrupted run could resume.

## Final measured results

| Measurement | Result |
|---|---:|
| One-time evaluation on the supplied 819-image internal test | **809 / 819 = 98.7790% top-1** |
| Final inference model file, `submission/model.pth` | **8.471 MB** |
| Final package CPU inference on our laptop, including image file read | **43.196 ms/image mean** |

The final-package timing used 50 warm-ups and 5 × 1,000 timed predictions. It was measured in a separate run from the fold-checkpoint comparison above, so the two timings should not be treated as interchangeable. Hardware, background load, and the organizer's preprocessing and timing method can change latency. **The 98.7790% result is from our supplied internal test, not the competition's unseen hidden test.** See [`EXPERIMENT_REPORT.md`](EXPERIMENT_REPORT.md) for all measured runs and timing statistics.

## Run the model on a CPU

The following commands use **Windows PowerShell** from the repository root. Install Python 3.12 first, then:

```powershell
git clone https://github.com/nisattm/flower-102-classifier.git
cd flower-102-classifier
py -3.12 -m venv .venv
./.venv/Scripts/python.exe -m pip install -r submission/requirements.txt
```

Replace the example path with an existing flower image:

```powershell
./.venv/Scripts/python.exe submission/predict.py --weights submission/model.pth --image "C:/Photos/flower.jpg" --device cpu
```

The command prints JSON containing a `class_id` from **1 to 102** (as a string), `class_name`, and softmax `confidence`. To process a folder recursively and write a CSV:

```powershell
./.venv/Scripts/python.exe submission/predict.py --weights submission/model.pth --folder "C:/Photos/flowers" --output predictions.csv --device cpu
```

For integration with an evaluator, `submission.predict.load_model("submission/model.pth", device="cpu")` returns a classifier with `predict(image_path)` and `logits(normalized_batch)` methods. `predict` performs the saved preprocessing and returns the original 1–102 class ID. The weights are stored in FP16 to reduce file size and loaded into FP32 for CPU computation. The model is instantiated with `pretrained=False`, so inference does not download weights. Package-specific instructions and a Python example are in [`submission/README.md`](submission/README.md). A different instructor-specified interface can call this API if required.

## Repository contents and reproducibility

| Location | Purpose |
|---|---|
| [`submission/`](submission/) | Final inference package: model, CPU prediction script, class names, requirements |
| [`EXPERIMENT_REPORT.md`](EXPERIMENT_REPORT.md) | Recorded training, validation, export, and benchmark results |
| [`FINAL_MODEL_DECISION.md`](FINAL_MODEL_DECISION.md) | Model-selection evidence and frozen training recipe |
| [`CPU_TARGET.md`](CPU_TARGET.md) | Interpretation of the competition's CPU scoring guideline |
| [`artifacts/data/folds.jsonl`](artifacts/data/folds.jsonl) | Frozen validation assignments |
| `train.py`, `train_final.py`, `evaluate_checkpoint.py` | Training and evaluation scripts |

The original image archive, pretrained-weight cache, and intermediate checkpoints are intentionally not in Git. Reproducing training requires the instructor-supplied `flower_data` archive; running inference requires only the `submission/` files and installed Python dependencies. The final training command was:

```powershell
./.venv/Scripts/python.exe train_final.py --data "C:/path/to/flower_data" --run artifacts/runs/efficient_final_all_dev --model efficient --batch 8 --workers 2 --epochs 14 --backbone-lr 0.0001
```

For later model versions, compare changes on the frozen development folds or genuinely new validation data. The supplied 819-image internal test was evaluated once after selection and should not be reused for repeated tuning.
