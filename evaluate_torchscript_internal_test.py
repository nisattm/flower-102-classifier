"""One explicitly unlocked internal-test check of the submitted model.pt.

This is the supplied 819-image test split, not the competition's hidden test.
The document specifies resize/crop but not interpolation; bilinear is assumed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from evaluate_torchscript_format import FoldImages


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("submission_ready/model.pt"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/data/manifest.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/locked_test_torchscript.json"))
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--unlock-test", action="store_true")
    args = parser.parse_args()
    if not args.unlock_test:
        parser.error("This evaluation requires --unlock-test")
    if args.output.exists():
        raise FileExistsError(f"An internal-test result already exists: {args.output}")

    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf8").splitlines()]
    rows = [row for row in rows if row["split"] == "test"]
    if len(rows) != 819 or any(not 1 <= row["class_id"] <= 102 for row in rows):
        raise ValueError("Unexpected supplied test split or class mapping")
    loader = DataLoader(FoldImages(args.data, rows, "evaluator"), batch_size=args.batch, num_workers=0)
    torch.set_num_threads(min(6, torch.get_num_threads()))
    model = torch.jit.load(str(args.model), map_location="cpu").eval()

    class_correct = [0] * 102
    class_count = [0] * 102
    started = time.perf_counter()
    with torch.inference_mode():
        for images, labels in loader:
            probabilities = model(images)
            if probabilities.shape != (len(labels), 102):
                raise ValueError(f"Wrong output shape: {tuple(probabilities.shape)}")
            if not torch.isfinite(probabilities).all():
                raise ValueError("Non-finite probabilities")
            if not torch.allclose(probabilities.sum(1), torch.ones(len(labels)), atol=1e-5, rtol=0):
                raise ValueError("Probability rows do not sum to one")
            predictions = probabilities.argmax(1)
            for label, prediction in zip(labels.tolist(), predictions.tolist()):
                class_count[label] += 1
                class_correct[label] += int(prediction == label)
    if any(count == 0 for count in class_count):
        raise ValueError("At least one class is missing from the internal test")
    correct = sum(class_correct)
    result = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": str(args.model),
        "model_sha256": sha256(args.model),
        "manifest_sha256": sha256(args.manifest),
        "split": "supplied_internal_test",
        "preprocessing": "RGB; resize short side to 256 with assumed bilinear interpolation; center crop 224; ToTensor 0-1; normalization inside model",
        "correct": correct,
        "total": len(rows),
        "top1_percent": correct / len(rows) * 100,
        "macro_top1_percent": sum(c / n for c, n in zip(class_correct, class_count)) / 102 * 100,
        "elapsed_seconds": time.perf_counter() - started,
        "cpu_threads": torch.get_num_threads(),
        "batch": args.batch,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
