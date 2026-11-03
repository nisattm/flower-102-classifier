"""Measure the one-file TorchScript submission's CPU forward latency."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("submission_ready/model.pt"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/benchmarks/efficient_final_torchscript_cpu.json"))
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--per-round", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    model = torch.jit.load(str(args.model), map_location="cpu").eval()
    image = torch.rand(1, 3, 224, 224)
    with torch.inference_mode():
        for _ in range(args.warmup):
            model(image)
        timings: list[float] = []
        round_means: list[float] = []
        for _ in range(args.rounds):
            current = []
            for _ in range(args.per_round):
                start = time.perf_counter_ns()
                model(image)
                current.append((time.perf_counter_ns() - start) / 1e6)
            timings.extend(current)
            round_means.append(statistics.mean(current))
    ordered = sorted(timings)
    result = {
        "model": str(args.model),
        "sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "bytes": args.model.stat().st_size,
        "device": "cpu",
        "input": "[1, 3, 224, 224] RGB float32 in [0, 1]",
        "scope": "TorchScript forward only; evaluator image preprocessing excluded",
        "threads": args.threads,
        "warmup": args.warmup,
        "rounds": args.rounds,
        "per_round": args.per_round,
        "mean_ms": statistics.mean(timings),
        "median_ms": statistics.median(timings),
        "p95_ms": ordered[int(0.95 * (len(ordered) - 1))],
        "round_means_ms": round_means,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
