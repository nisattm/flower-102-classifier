# Final model decision — 26 October 2026

The model choice and training recipe are frozen before opening the provided 819-image internal locked test. This test is separate from the competition's hidden test.

| Candidate | Correct / 7,339 three-fold images | Top-1 | FP16 file | Local CPU disk-included mean (50 warmups, 5 × 1,000) |
|---|---:|---:|---:|---:|
| EfficientNet-B0, backbone LR 1e-4 | 7,226 | 98.4603% | 8.477 MB | 30.210 ms/image |
| MobileNetV4 Conv Small, backbone LR 3e-4 | 7,184 | 97.8880% | 5.391 MB | 18.073 ms/image |

Both local measurements satisfy the guideline's **proposed**, unconfirmed 50 ms and 35 MB thresholds. With its stated saturated score formula, their projected local scores are 99.0762 and 98.7328 respectively. These are not competition scores: the professor's CPU model and exact timing method are unknown. EfficientNet-B0 is selected because its three-fold accuracy is 42 more correct images and its local timing has about 19.8 ms of margin to the proposed limit.

Frozen final training recipe: `efficientnet_b0.ra4_e3600_r224_in1k`, pretrained weights already used for three-fold validation, 224 × 224, seed 42, batch 8, effective batch 32, AdamW weight decay 0.05, label smoothing 0.1, first 3 head-only epochs, backbone LR 1e-4, head LR 1e-3, cosine decay, **14 total epochs**. The 14-epoch choice follows the best epochs 14, 14, and 12 in the three folds. Train on all 7,339 unique images in the frozen development folds only. Save at each epoch for resume. No validation metric will be claimed for this full-development model. Export FP16 weights for CPU loading as FP32, verify offline inference, then evaluate once on the internal locked test.

Evidence: `artifacts/runs/{efficient_f0_v5,efficient_f1,efficient_f2,mobile_lr3e4_f0,mobile_lr3e4_f1,mobile_lr3e4_f2}/verified_best.json`, `artifacts/benchmarks/{efficient_f0_v5,mobile_lr3e4_f0}_fp16_cpu_full.json`, and `EXPERIMENT_REPORT.md`.
