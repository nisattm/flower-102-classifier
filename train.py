"""Resumable 102-class fine tuning. The original test split is never read here."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T


MODELS = {
    'mobile': 'mobilenetv4_conv_small.e2400_r224_in1k',
    'efficient': 'efficientnet_b0.ra4_e3600_r224_in1k',
    'teacher': 'convnextv2_tiny.fcmae_ft_in22k_in1k',
}
class Flowers(Dataset):
    def __init__(self, root: Path, rows: list[dict], size: int, training: bool, data_config: dict):
        self.root, self.rows = root, rows
        mean, std = data_config['mean'], data_config['std']
        interpolation = T.InterpolationMode.BICUBIC if data_config['interpolation'] == 'bicubic' else T.InterpolationMode.BILINEAR
        fill = tuple(round(float(v) * 255) for v in mean)
        self.transform = (T.Compose([
            T.RandomResizedCrop(size, scale=(0.6, 1.0), interpolation=interpolation), T.RandomHorizontalFlip(),
            T.RandomRotation(30, interpolation=interpolation, fill=fill),
            T.ColorJitter(brightness=.12, contrast=.12, saturation=.08, hue=.02),
            T.ToTensor(), T.Normalize(mean, std), T.RandomErasing(p=.1, scale=(.02, .08)),
        ]) if training else T.Compose([
            T.Resize(round(size / data_config['crop_pct']), interpolation=interpolation), T.CenterCrop(size),
            T.ToTensor(), T.Normalize(mean, std),
        ]))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(self.root / row['path']) as im:
            image = ImageOps.exif_transpose(im).convert('RGB')
        return self.transform(image), row['class_id'] - 1


def atomic_save(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state() -> dict:
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state: dict) -> None:
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def evaluate(model, loader, device, amp: bool) -> dict:
    model.eval()
    dtype = next(model.parameters()).dtype
    cm = torch.zeros((102, 102), dtype=torch.int64)
    loss_sum, count, amp_fallback_batches = 0., 0, 0
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device, dtype=dtype), y.to(device)
            with torch.autocast(device_type='cuda', enabled=amp):
                logits = model(x)
                loss = nn.functional.cross_entropy(logits.float(), y)
            if amp and not torch.isfinite(loss):
                # Some EfficientNet validation images overflow in CUDA FP16.
                # Retry only that batch in FP32; reject genuinely bad weights.
                with torch.autocast(device_type='cuda', enabled=False):
                    logits = model(x)
                    loss = nn.functional.cross_entropy(logits.float(), y)
                amp_fallback_batches += 1
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite validation loss')
            pred = logits.argmax(1)
            bins = torch.bincount((y * 102 + pred).cpu(), minlength=102 * 102)
            cm += bins.reshape(102, 102)
            loss_sum += float(loss) * len(y)
            count += len(y)
    per_class = cm.diag() / cm.sum(1).clamp(min=1)
    return dict(loss=loss_sum / count, top1=float(cm.diag().sum() / count),
                macro=float(per_class.mean()), confusion=cm.tolist(), count=count,
                amp_fallback_batches=amp_fallback_batches)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--folds', type=Path, default=Path('artifacts/data/folds.jsonl'))
    p.add_argument('--manifest', type=Path, default=Path('artifacts/data/manifest.jsonl'))
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--model', choices=MODELS, default='efficient')
    p.add_argument('--fold', type=int, choices=(0, 1, 2), default=0)
    p.add_argument('--size', type=int, default=224)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--effective-batch', type=int, default=32)
    p.add_argument('--head-epochs', type=int, default=3)
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--backbone-lr', type=float, default=1e-4)
    p.add_argument('--head-lr', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--patience', type=int, default=0, help='Stop after this many epochs without top-1 improvement; 0 disables')
    p.add_argument('--mixup-alpha', type=float, default=0., help='MixUp alpha; 0 disables')
    args = p.parse_args()
    if args.effective_batch % args.batch:
        p.error('--effective-batch must be divisible by --batch')
    args.run.mkdir(parents=True, exist_ok=True)
    fold_bytes = args.folds.read_bytes()
    fold_hash = hashlib.sha256(fold_bytes).hexdigest()
    rows = [json.loads(line) for line in fold_bytes.splitlines()]
    train_rows = [r for r in rows if r['fold'] != args.fold]
    val_rows = [r for r in rows if r['fold'] == args.fold]
    seed_all(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is unavailable; select a GPU environment before training')
    device = torch.device('cuda')
    amp = True
    model = timm.create_model(MODELS[args.model], pretrained=not (args.run / 'last.pt').exists(), num_classes=102)
    data_config = timm.data.resolve_model_data_config(model)
    manifest_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    config = dict(model=MODELS[args.model], fold=args.fold, fold_sha256=fold_hash,
                  manifest_sha256=manifest_hash, size=args.size, batch=args.batch,
                  effective_batch=args.effective_batch, head_epochs=args.head_epochs,
                  epochs=args.epochs, backbone_lr=args.backbone_lr, head_lr=args.head_lr,
                  seed=args.seed, mean=data_config['mean'], std=data_config['std'],
                  interpolation=data_config['interpolation'], crop_pct=data_config['crop_pct'],
                  train_count=len(train_rows), val_count=len(val_rows))
    if args.patience:
        config['patience'] = args.patience
    if args.mixup_alpha:
        config['mixup_alpha'] = args.mixup_alpha
    class_names = json.loads(Path('class_names.json').read_text(encoding='utf8'))
    if set(class_names) != {str(i) for i in range(1, 103)}:
        raise ValueError('class_names.json must contain IDs 1..102')
    cfg_path = args.run / 'config.json'
    if cfg_path.exists():
        if json.loads(cfg_path.read_text()) != json.loads(json.dumps(config)):
            raise ValueError('Existing run has different config or data; choose a new --run')
    else:
        cfg_path.write_text(json.dumps(config, indent=2), encoding='utf8')
    model.to(device)
    head = list(model.get_classifier().parameters())
    head_ids = {id(x) for x in head}
    backbone = [x for x in model.parameters() if id(x) not in head_ids]
    optimizer = torch.optim.AdamW([{'params': backbone, 'lr': 0.},
                                   {'params': head, 'lr': args.head_lr}], weight_decay=.05)
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    best, start, bad_epochs = -1., 0, 0
    last = args.run / 'last.pt'
    if last.exists():
        ckpt = torch.load(last, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scaler.load_state_dict(ckpt['scaler'])
        restore_rng(ckpt['rng'])
        best, start, bad_epochs = ckpt['best_top1'], ckpt['epoch'] + 1, ckpt.get('bad_epochs', 0)
        print(f'Resuming from epoch {start + 1}', flush=True)
    train_ds = Flowers(args.data, train_rows, args.size, True, data_config)
    val_ds = Flowers(args.data, val_rows, args.size, False, data_config)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                            pin_memory=amp)
    accum = args.effective_batch // args.batch
    for epoch in range(start, args.epochs):
        # Freeze backbone for warmup, then use cosine learning-rate decay.
        full = epoch >= args.head_epochs
        decay = .5 * (1 + math.cos(math.pi * (epoch - args.head_epochs) /
                                     max(1, args.epochs - args.head_epochs))) if full else 1.
        optimizer.param_groups[0]['lr'] = args.backbone_lr * decay if full else 0.
        optimizer.param_groups[1]['lr'] = args.head_lr * decay
        for param in backbone:
            param.requires_grad_(full)
        # Epoch-specific order is independent of data-loader RNG state after restart.
        generator = torch.Generator().manual_seed(args.seed + epoch)
        train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, generator=generator,
                                  num_workers=args.workers, pin_memory=amp)
        model.train() if full else model.eval()
        optimizer.zero_grad(set_to_none=True)
        t0, train_loss, seen = time.perf_counter(), 0., 0
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            if args.mixup_alpha:
                lam = float(np.random.beta(args.mixup_alpha, args.mixup_alpha))
                shuffled = torch.randperm(len(y), device=device)
                x = lam * x + (1 - lam) * x[shuffled]
            with torch.autocast(device_type='cuda', enabled=amp):
                logits = model(x)
                if args.mixup_alpha:
                    loss = (lam * nn.functional.cross_entropy(logits.float(), y, label_smoothing=.1)
                            + (1 - lam) * nn.functional.cross_entropy(logits.float(), y[shuffled], label_smoothing=.1))
                else:
                    loss = nn.functional.cross_entropy(logits.float(), y, label_smoothing=.1)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite training loss at epoch {epoch + 1}, step {step + 1}')
            scaler.scale(loss / accum).backward()
            if (step + 1) % accum == 0 or step + 1 == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            train_loss += float(loss.detach()) * len(y)
            seen += len(y)
        metrics = evaluate(model, val_loader, device, amp)
        duration = time.perf_counter() - t0
        summary = dict(epoch=epoch + 1, train_loss=train_loss / seen,
                       val_loss=metrics['loss'], top1=metrics['top1'], macro=metrics['macro'],
                       seconds=duration, batch=args.batch, workers=args.workers,
                       peak_cuda_mb=(torch.cuda.max_memory_allocated() / 1e6 if amp else None))
        if metrics['top1'] > best:
            best = metrics['top1']
            bad_epochs = 0
            weights = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            atomic_save(dict(model=weights, config=config, class_names=class_names, epoch=epoch + 1,
                             top1=best), args.run / 'best.pt')
            (args.run / 'best_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf8')
        else:
            bad_epochs += 1
        atomic_save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                         scaler=scaler.state_dict(), rng=rng_state(), epoch=epoch,
                         best_top1=best, bad_epochs=bad_epochs), last)
        with (args.run / 'history.jsonl').open('a', encoding='utf8') as f:
            f.write(json.dumps(summary) + '\n')
        print(json.dumps(summary), flush=True)
        if args.patience and bad_epochs >= args.patience:
            (args.run / 'early_stop.json').write_text(json.dumps(dict(epoch=epoch + 1,
                bad_epochs=bad_epochs, best_top1=best), indent=2), encoding='utf8')
            print(f'Early stopping after {bad_epochs} epochs without improvement', flush=True)
            break


if __name__ == '__main__':
    main()
