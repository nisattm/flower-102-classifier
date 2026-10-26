"""Evaluate a saved model on its development fold; test requires an explicit unlock flag."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from predict import load_model
from train import Flowers, evaluate


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--folds', type=Path, default=Path('artifacts/data/folds.jsonl'))
    p.add_argument('--manifest', type=Path, default=Path('artifacts/data/manifest.jsonl'))
    p.add_argument('--split', choices=('fold', 'test'), default='fold')
    p.add_argument('--unlock-test', action='store_true')
    p.add_argument('--device', default='cpu')
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    if args.split == 'test' and not args.unlock_test:
        p.error('The test split is locked. Use --unlock-test only after finalist settings are frozen.')
    classifier = load_model(args.weights, args.device)
    cfg = classifier.config
    if args.split == 'fold':
        data = args.folds.read_bytes()
        if hashlib.sha256(data).hexdigest() != cfg['fold_sha256']:
            raise ValueError('Fold file differs from checkpoint training folds')
        rows = [json.loads(line) for line in data.splitlines() if json.loads(line)['fold'] == cfg['fold']]
    else:
        rows = [json.loads(line) for line in args.manifest.read_bytes().splitlines()
                if json.loads(line)['split'] == 'test']
    ds = Flowers(args.data, rows, cfg['size'], False, cfg)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=0,
                        pin_memory=args.device.startswith('cuda'))
    metrics = evaluate(classifier.model, loader, torch.device(args.device), args.device.startswith('cuda'))
    metrics.update(split=args.split, checkpoint=str(args.weights), epoch=cfg.get('epoch'))
    if args.output:
        args.output.write_text(json.dumps(metrics, indent=2), encoding='utf8')
    print(json.dumps({key: metrics[key] for key in ('split', 'count', 'top1', 'macro', 'loss')}, indent=2))


if __name__ == '__main__':
    main()
