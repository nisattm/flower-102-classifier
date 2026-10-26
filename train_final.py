"""Train the frozen finalist on all development images, without reading the locked test."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import timm
import torch
from torch import nn
from torch.utils.data import DataLoader

from train import MODELS, Flowers, atomic_save, restore_rng, rng_state, seed_all


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--folds', type=Path, default=Path('artifacts/data/folds.jsonl'))
    p.add_argument('--manifest', type=Path, default=Path('artifacts/data/manifest.jsonl'))
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--model', choices=MODELS, default='efficient')
    p.add_argument('--size', type=int, default=224)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--effective-batch', type=int, default=32)
    p.add_argument('--head-epochs', type=int, default=3)
    p.add_argument('--epochs', type=int, required=True)
    p.add_argument('--backbone-lr', type=float, default=1e-4)
    p.add_argument('--head-lr', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--workers', type=int, default=2)
    args = p.parse_args()
    if args.effective_batch % args.batch or args.epochs <= args.head_epochs:
        p.error('effective batch must be divisible by batch; epochs must exceed head epochs')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is unavailable')

    fold_bytes = args.folds.read_bytes()
    rows = [json.loads(line) for line in fold_bytes.splitlines()]
    paths = [row['path'] for row in rows]
    if len(paths) != len(set(paths)) or not rows or any(Path(path).parts[0].lower() == 'test' for path in paths):
        raise ValueError('Frozen development folds contain duplicate paths, test paths, or no images')
    if set(row['fold'] for row in rows) != {0, 1, 2}:
        raise ValueError('Expected the frozen three-fold development set')
    if any(not 1 <= row['class_id'] <= 102 for row in rows):
        raise ValueError('Invalid class ID in frozen development folds')
    class_names = json.loads(Path('class_names.json').read_text(encoding='utf8'))
    if set(class_names) != {str(i) for i in range(1, 103)}:
        raise ValueError('class_names.json must contain IDs 1..102')

    args.run.mkdir(parents=True, exist_ok=True)
    last = args.run / 'last.pt'
    seed_all(args.seed)
    device = torch.device('cuda')
    model = timm.create_model(MODELS[args.model], pretrained=not last.exists(), num_classes=102)
    data_config = timm.data.resolve_model_data_config(model)
    config = dict(model=MODELS[args.model], fold='all_development',
                  fold_sha256=hashlib.sha256(fold_bytes).hexdigest(),
                  manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                  size=args.size, batch=args.batch, effective_batch=args.effective_batch,
                  head_epochs=args.head_epochs, epochs=args.epochs,
                  backbone_lr=args.backbone_lr, head_lr=args.head_lr, seed=args.seed,
                  mean=data_config['mean'], std=data_config['std'],
                  interpolation=data_config['interpolation'], crop_pct=data_config['crop_pct'],
                  train_count=len(rows), selection='frozen three-fold CPU score')
    cfg_path = args.run / 'config.json'
    serialized = json.dumps(config, indent=2)
    if cfg_path.exists():
        if json.loads(cfg_path.read_text(encoding='utf8')) != json.loads(serialized):
            raise ValueError('Existing final run has different config or data')
    else:
        cfg_path.write_text(serialized, encoding='utf8')

    model.to(device)
    head = list(model.get_classifier().parameters())
    head_ids = {id(param) for param in head}
    backbone = [param for param in model.parameters() if id(param) not in head_ids]
    optimizer = torch.optim.AdamW([{'params': backbone, 'lr': 0.},
                                   {'params': head, 'lr': args.head_lr}], weight_decay=.05)
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    start = 0
    if last.exists():
        ckpt = torch.load(last, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scaler.load_state_dict(ckpt['scaler'])
        restore_rng(ckpt['rng'])
        start = ckpt['epoch'] + 1
        print(f'Resuming final training from epoch {start + 1}', flush=True)

    dataset = Flowers(args.data, rows, args.size, True, data_config)
    accum = args.effective_batch // args.batch
    for epoch in range(start, args.epochs):
        full = epoch >= args.head_epochs
        decay = (.5 * (1 + math.cos(math.pi * (epoch - args.head_epochs) /
                                  max(1, args.epochs - args.head_epochs))) if full else 1.)
        optimizer.param_groups[0]['lr'] = args.backbone_lr * decay if full else 0.
        optimizer.param_groups[1]['lr'] = args.head_lr * decay
        for param in backbone:
            param.requires_grad_(full)
        generator = torch.Generator().manual_seed(args.seed + epoch)
        loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, generator=generator,
                            num_workers=args.workers, pin_memory=True)
        model.train() if full else model.eval()
        optimizer.zero_grad(set_to_none=True)
        t0, train_loss, seen = time.perf_counter(), 0., 0
        for step, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type='cuda'):
                logits = model(x)
                loss = nn.functional.cross_entropy(logits.float(), y, label_smoothing=.1)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite final training loss at epoch {epoch + 1}, step {step + 1}')
            scaler.scale(loss / accum).backward()
            if (step + 1) % accum == 0 or step + 1 == len(loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            train_loss += float(loss.detach()) * len(y)
            seen += len(y)
        atomic_save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                         scaler=scaler.state_dict(), rng=rng_state(), epoch=epoch), last)
        summary = dict(epoch=epoch + 1, train_loss=train_loss / seen,
                       seconds=time.perf_counter() - t0, train_count=seen,
                       peak_cuda_mb=torch.cuda.max_memory_allocated() / 1e6)
        with (args.run / 'history.jsonl').open('a', encoding='utf8') as stream:
            stream.write(json.dumps(summary) + '\n')
        print(json.dumps(summary), flush=True)

    if start < args.epochs or not (args.run / 'final.pt').exists():
        weights = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        atomic_save(dict(model=weights, config=config, class_names=class_names,
                         epoch=args.epochs), args.run / 'final.pt')
        print(f'Wrote {args.run / "final.pt"}', flush=True)


if __name__ == '__main__':
    main()
