"""Check a TorchScript model using the documented evaluator resize/crop.

The PDF does not specify interpolation, so this check assumes bilinear resize.
Only frozen development-fold images are read. The internal test is not touched.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T


class FoldImages(Dataset):
    def __init__(self, root: Path, rows: list[dict], preprocessing: str):
        self.root = root
        self.rows = rows
        if preprocessing == "evaluator":
            self.transform = T.Compose([
                T.Resize(256, interpolation=T.InterpolationMode.BILINEAR),
                T.CenterCrop(224),
                T.ToTensor(),
            ])
        else:
            self.transform = T.Compose([
                T.Resize(249, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(224),
                T.ToTensor(),
            ])
        self.preprocessing = preprocessing

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(self.root / row["path"]) as image:
            if self.preprocessing == "training":
                image = ImageOps.exif_transpose(image)
            tensor = self.transform(image.convert("RGB"))
        return tensor, row["class_id"] - 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--folds", type=Path, default=Path("artifacts/data/folds.jsonl"))
    parser.add_argument("--fold", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--preprocessing", choices=("evaluator", "training"), default="evaluator")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.folds.read_text(encoding="utf8").splitlines()]
    rows = [row for row in rows if row["fold"] == args.fold]
    if not rows:
        raise ValueError("No images in the requested frozen fold")
    dataset = FoldImages(args.data, rows, args.preprocessing)
    loader = DataLoader(dataset, batch_size=args.batch, num_workers=0)
    torch.set_num_threads(min(6, torch.get_num_threads()))
    model = torch.jit.load(str(args.model), map_location="cpu").eval()

    correct = 0
    total = 0
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
            correct += int((probabilities.argmax(1) == labels).sum())
            total += len(labels)
    result = {
        "model": str(args.model),
        "fold": args.fold,
        "preprocessing": args.preprocessing,
        "correct": correct,
        "total": total,
        "top1_percent": correct / total * 100,
        "elapsed_seconds": time.perf_counter() - started,
        "cpu_threads": torch.get_num_threads(),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
