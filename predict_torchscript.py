"""Local image demo for model.pt; this script is not part of the submitted file."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms as T


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("submission_ready/model.pt"))
    parser.add_argument("--names", type=Path, default=Path("class_names.json"))
    args = parser.parse_args()

    transform = T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(224),
        T.ToTensor(),
    ])
    with Image.open(args.image) as image:
        tensor = transform(image.convert("RGB")).unsqueeze(0)
    model = torch.jit.load(str(args.model), map_location="cpu").eval()
    with torch.inference_mode():
        probabilities = model(tensor)[0]
    class_id = int(probabilities.argmax()) + 1
    names = json.loads(args.names.read_text(encoding="utf8"))
    print(json.dumps({
        "class_id": class_id,
        "class_name": names[str(class_id)],
        "confidence": float(probabilities[class_id - 1]),
    }, indent=2))


if __name__ == "__main__":
    main()
