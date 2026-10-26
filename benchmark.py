"""Measure batch-one prediction latency and actual weight-file size."""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from predict import load_model


def stats(values: list[float]) -> dict:
    return dict(mean_ms=statistics.mean(values), median_ms=statistics.median(values),
                p95_ms=float(np.percentile(values, 95)), n=len(values))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--image', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--warmup', type=int, default=50)
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--per-repeat', type=int, default=1000)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    t0 = time.perf_counter()
    classifier = load_model(args.weights, args.device)
    load_seconds = time.perf_counter() - t0
    with Image.open(args.image) as im:
        opened = ImageOps.exif_transpose(im).convert('RGB')
    dtype = next(classifier.model.parameters()).dtype
    tensor = classifier.transform(opened).unsqueeze(0).to(args.device, dtype=dtype)

    def sync():
        if args.device.startswith('cuda'):
            torch.cuda.synchronize()

    @torch.inference_mode()
    def run_one(kind: str):
        if kind == 'tensor':
            classifier.model(tensor)
        elif kind == 'opened':
            x = classifier.transform(opened).unsqueeze(0).to(args.device, dtype=dtype)
            classifier.model(x)
        else:
            with Image.open(args.image) as im:
                x = classifier.transform(ImageOps.exif_transpose(im).convert('RGB')).unsqueeze(0).to(args.device, dtype=dtype)
            classifier.model(x)
        sync()

    results = {}
    for kind in ('tensor', 'opened', 'disk'):
        for _ in range(args.warmup):
            run_one(kind)
        times = []
        for _ in range(args.repeats):
            for _ in range(args.per_repeat):
                sync()
                start = time.perf_counter()
                run_one(kind)
                times.append((time.perf_counter() - start) * 1000)
        results[kind] = stats(times)
    report = dict(device=args.device, cpu=platform.processor(), gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
                  torch=torch.__version__, threads=torch.get_num_threads(), weights_mb=args.weights.stat().st_size / 1e6,
                  load_seconds=load_seconds, warmup=args.warmup, repeats=args.repeats,
                  per_repeat=args.per_repeat, latency=results)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
