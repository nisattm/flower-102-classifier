# Flower 102 classification

This project classifies the 102 categories in Oxford Flowers 102. The selected model is EfficientNet-B0, trained on the supplied flower archive and packaged for single-image CPU inference. The dataset itself is **not** included in the repository.

## Try the model

Use Python 3.12 and install the packages in [`submission/requirements.txt`](submission/requirements.txt). PyTorch's CPU build is sufficient for inference. Then run:

```powershell
python submission/predict.py --weights submission/model.pth --image path\to\flower.jpg --device cpu
python submission/predict.py --weights submission/model.pth --folder path\to\photos --output predictions.csv --device cpu
```

The result contains a `class_id` from **1–102**, a `class_name`, and a confidence value. In Python, `from predict import load_model` inside `submission/` provides `classifier.predict(path)` and `classifier.logits(normalized_tensor)`. The inference code loads the package without downloading pretrained weights. See [`submission/README.md`](submission/README.md) for the small standalone package.

## Measured results

| Measurement | Result |
|---|---:|
| Three-fold development top-1 | 7,226 / 7,339 = **98.4603%** |
| One-time internal locked test top-1 | 809 / 819 = **98.7790%** |
| Final FP16 model file | **8.471 MB** |
| Final FP16 CPU latency, including file read | **43.196 ms/image** on this laptop, 50 warmups + 5 × 1,000 timings |
| Candidate FP16 CPU latency, including file read | **30.210 ms/image** on this laptop, 50 warmups + 5 × 1,000 timings |

The final package and candidate were measured in separate runs; the difference shows that local timing varies. Both local means are below the proposed 50 ms limit. Full latency statistics are in [`EXPERIMENT_REPORT.md`](EXPERIMENT_REPORT.md). The internal locked test is the supplied 819-image `test` split; it is separate from the competition's unseen test. The organizer's CPU is an Intel Core i7 with 16 GB RAM and no GPU. The guideline gives a score of 60% accuracy, 25% CPU speed and 15% file size; 50 ms and 35 MB are proposed, not confirmed, thresholds. Local timing is not the organizer's timing.

## Training and evaluation

The original archive had 8,189 images: 6,552 `train`, 818 `valid` and 819 `test`. Training and validation used frozen three-fold splits made from `train + valid` after removing confirmed duplicates and test-observation overlaps. The final model was trained for 14 epochs on all 7,339 remaining development images. It never trained on the 819 test images. See [`FINAL_MODEL_DECISION.md`](FINAL_MODEL_DECISION.md), [`CPU_TARGET.md`](CPU_TARGET.md) and [`EXPERIMENT_REPORT.md`](EXPERIMENT_REPORT.md) for the exact selection rule, CPU protocol, run results and data audit.

Training is resumable at each completed epoch. A matching command resumes from `last.pt` in the same run directory; a changed configuration or fold file is rejected. The final training command was:

```powershell
python train_final.py --data path\to\flower_data --run artifacts\runs\efficient_final_all_dev --model efficient --batch 8 --workers 2 --epochs 14 --backbone-lr 0.0001
```

Auxiliary scripts read `FLOWER_DATA_ROOT` when set; otherwise they expect a `flower_data` directory in the project. The inference package does not need the training images.

The `artifacts/runs/` checkpoints, pretrained caches, original images, and old model exports are intentionally omitted from Git. The deployable `submission/model.pth` is included. [`FLOWER_PLAN.txt`](FLOWER_PLAN.txt) is the original plan; later competition details and measured results are reflected in the decision and report files.

## Future experiments

Accuracy and CPU speed can be improved in later versions. Use the same frozen development folds or a genuinely new validation set to compare changes, then publish a new model version. The 819-image internal test was used once after this version's settings were frozen and should not be repeatedly used to tune later versions. The organizer's hidden test remains unseen.
