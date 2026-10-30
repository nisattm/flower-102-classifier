# Flower 102 — CPU Inference Package

This directory is the **ready-to-run submission** for the 102-class flower recognition project. It contains the selected EfficientNet-B0 model, its preprocessing and class mapping, and a Python inference script. The model runs on a CPU without the training dataset or a GPU. Once Python dependencies are installed, inference does not require a network connection.

**Project team:** Nisa Atım (220205014), Ceren Yüksel (220201039), Mustafa Eymen Aslan (210201004), and Efekan Coşkun (220201024).

For the dataset audit, model comparison, training recipe, and interpretation of results, see the [main project README](../README.md) and [experiment report](../EXPERIMENT_REPORT.md).

## Included files

| File | Purpose |
|---|---|
| `model.pth` | Final EfficientNet-B0 weights, preprocessing configuration, and embedded class names |
| `predict.py` | Single-image, recursive folder, and Python API inference |
| `class_names.json` | Human-readable names for the original class IDs 1–102 |
| `requirements.txt` | Pinned Python package versions |

Keep these files together. The saved `model.pth` is approximately **8.47 MB**. It stores weights in FP16 for a smaller file; the code loads them into FP32 for CPU computation.

## Installation

Use **Python 3.12** and run the following commands **inside this `submission` directory**. On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt
```

On macOS or Linux, if compatible PyTorch packages are available:

```bash
python3.12 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
```

Package installation needs access to the Python package index. CPU inference was verified on Windows; other operating systems may require compatible package builds.

## Predict one image

On Windows, replace the path below with an existing photograph. Quote paths that contain spaces:

```powershell
./.venv/Scripts/python.exe predict.py --weights model.pth --image "C:/Photos/flower.jpg" --device cpu
```

On macOS or Linux, use `./.venv/bin/python` in the same command. The result is JSON in this format:

```json
{
  "class_id": "1",
  "class_name": "pink primrose",
  "confidence": 0.9916
}
```

`class_id` is the **original 1–102 category number, returned as a string**; it is not a zero-based index. `class_name` is the corresponding readable label. `confidence` is the predicted class's softmax score, not a guarantee that the classification is correct. The classifier will still choose one of the 102 classes for an out-of-distribution image.

## Predict a folder

```powershell
./.venv/Scripts/python.exe predict.py --weights model.pth --folder "C:/Photos/flowers" --output predictions.csv --device cpu
```

The script searches subdirectories and accepts `.jpg`, `.jpeg`, `.png`, and `.webp` files. The CSV columns are `path`, `class_id`, `class_name`, and `confidence`. Omit `--output` to print a JSON list instead. The output directory must already exist.

## Python API and evaluator integration

```python
from predict import load_model

classifier = load_model("model.pth", device="cpu")
result = classifier.predict("flower.jpg")
class_id = int(result["class_id"])  # Original class ID, 1 through 102
```

`classifier.predict(path)` opens the image, corrects EXIF orientation, converts it to RGB, and applies the model's saved resize, crop, and normalization configuration. `classifier.logits(images)` is also available for an evaluator that already prepares normalized `[batch, 3, height, width]` tensors; it returns raw `[batch, 102]` scores. The command-line `--names path/to/class_names.json` option can override the embedded names.

The model is created with `pretrained=False` and loaded from the included file, so no weights are downloaded during inference. If an instructor provides a different required function signature, a small adapter can call `load_model` and return the same 1–102 class ID.

## What the measured numbers mean

The final model was trained for **14 epochs on 7,339 cleaned development images**. After model selection and settings were frozen, it classified **809 of 819 supplied internal-test images correctly (98.7790% top-1)**. This internal test is separate from the competition's hidden test. On our laptop, the final package averaged **43.196 ms per image on CPU including file read** in a benchmark with 50 warm-ups and five rounds of 1,000 predictions. The instructor's CPU and timing procedure may produce a different result.

If a command fails, check that it is run from this directory, the image and `model.pth` paths exist, and the packages from `requirements.txt` were installed into the Python environment used for the command.
