"""Offline single-image and directory inference from a lightweight best.pt file."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import timm
import torch
from PIL import Image, ImageOps
from torchvision import transforms as T


class Classifier:
    def __init__(self, model, config: dict, device: str, names: dict[str, str] | None = None):
        self.model = model.eval()
        self.config = config
        self.device = torch.device(device)
        self.names = names or {}
        size = config['size']
        self.transform = T.Compose([
            T.Resize(round(size / config['crop_pct']), interpolation=(T.InterpolationMode.BICUBIC if config['interpolation'] == 'bicubic' else T.InterpolationMode.BILINEAR)),
            T.CenterCrop(size), T.ToTensor(),
            T.Normalize(config['mean'], config['std']),
        ])

    @torch.inference_mode()
    def logits(self, images: torch.Tensor) -> torch.Tensor:
        """[B, 3, H, W] -> [B, 102] raw logits, already normalized input."""
        dtype = next(self.model.parameters()).dtype
        return self.model(images.to(self.device, dtype=dtype))

    def predict(self, path: str | Path) -> dict:
        with Image.open(path) as im:
            x = self.transform(ImageOps.exif_transpose(im).convert('RGB')).unsqueeze(0)
        probabilities = self.logits(x).softmax(1)[0]
        index = int(probabilities.argmax())
        class_id = index + 1
        return {'class_id': str(class_id), 'class_name': self.names.get(str(class_id), f'class_{class_id}'),
                'confidence': float(probabilities[index])}


def load_model(path: str | Path, device: str = 'cpu', names_path: str | Path | None = None) -> Classifier:
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    config = ckpt['config']
    model = timm.create_model(config['model'], pretrained=False, num_classes=102)
    model.load_state_dict(ckpt['model'])
    if ckpt.get('precision') == 'fp16' and str(device).startswith('cuda'):
        model.half()
    model.to(device)
    names = json.loads(Path(names_path).read_text(encoding='utf8')) if names_path else ckpt.get('class_names', {})
    return Classifier(model, config, device, names)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--image', type=Path)
    p.add_argument('--folder', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--device', default='cpu')
    p.add_argument('--names', type=Path)
    args = p.parse_args()
    classifier = load_model(args.weights, args.device, args.names)
    if args.image:
        print(json.dumps(classifier.predict(args.image), ensure_ascii=False, indent=2))
    if args.folder:
        paths = sorted(p for p in args.folder.rglob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'})
        rows = [dict(path=str(path), **classifier.predict(path)) for path in paths]
        if args.output:
            with args.output.open('w', newline='', encoding='utf8') as f:
                writer = csv.DictWriter(f, fieldnames=['path', 'class_id', 'class_name', 'confidence'])
                writer.writeheader()
                writer.writerows(rows)
        else:
            print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
