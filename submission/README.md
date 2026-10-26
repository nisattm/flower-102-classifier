# Flower 102 CPU submission

This folder contains one EfficientNet-B0 classifier for the 102 flower classes. The returned `class_id` is the original numeric category **1–102**. `class_name` comes from `class_names.json` embedded in the model package.

Install the versions in `requirements.txt` in a Python 3.12 environment, then run on a CPU:

```powershell
python predict.py --weights model.pth --image path\to\flower.jpg --device cpu
python predict.py --weights model.pth --folder path\to\images --output predictions.csv --device cpu
```

The image result is JSON with `class_id`, `class_name`, and `confidence`. Directory mode writes CSV with one row per image. The model loads with `pretrained=False` and does not download weights at inference time. `model.pth` stores FP16 weights for file size; `predict.py` loads them into FP32 on CPU.

The model was selected using three development folds. In one evaluation after selection, it classified 809 of the provided 819 internal-test images correctly (98.7790% top-1). This is separate from the competition's hidden test. Hidden-test accuracy and timing are determined by the organizer's CPU environment.

The final package measured 43.196 ms/image including disk read on the local Intel CPU (50 warmups, five rounds of 1,000). The selected fold model measured 30.210 ms/image in an earlier run on the same laptop, so timing varies with the local environment. The organizer's exact CPU and timing method are unknown.
