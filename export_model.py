"""Strip a best checkpoint to an offline deployable FP32 or FP16 weight file."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--precision', choices=('fp32', 'fp16'), default='fp32')
    args = p.parse_args()
    source = torch.load(args.weights, map_location='cpu', weights_only=False)
    names = source.get('class_names') or json.loads(Path('class_names.json').read_text(encoding='utf8'))
    weights = {name: (tensor.half() if args.precision == 'fp16' and tensor.is_floating_point() else tensor)
               for name, tensor in source['model'].items()}
    package = dict(model=weights, config=source['config'], class_names=names,
                   precision=args.precision, source_epoch=source['epoch'],
                   development_top1=source.get('top1'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + '.tmp')
    torch.save(package, temp)
    os.replace(temp, args.output)
    print(json.dumps(dict(output=str(args.output), precision=args.precision,
                          bytes=args.output.stat().st_size, mb=args.output.stat().st_size / 1e6), indent=2))


if __name__ == '__main__':
    main()
