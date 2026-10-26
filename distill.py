"""Resumeable teacher-to-student training on one frozen development fold."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import timm
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from train import Flowers, atomic_save, evaluate, restore_rng, rng_state, seed_all


class SharedAugmentFlowers(Dataset):
    def __init__(self, root: Path, rows: list[dict], size: int, interpolation: str):
        self.root, self.rows = root, rows
        mode = T.InterpolationMode.BICUBIC if interpolation == 'bicubic' else T.InterpolationMode.BILINEAR
        self.transform = T.Compose([
            T.RandomResizedCrop(size, scale=(.6, 1.), interpolation=mode),
            T.RandomHorizontalFlip(),
            T.RandomRotation(30, interpolation=mode, fill=(127, 127, 127)),
            T.ColorJitter(brightness=.12, contrast=.12, saturation=.08, hue=.02),
            T.ToTensor(), T.RandomErasing(p=.1, scale=(.02, .08), value=.5),
        ])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(self.root / row['path']) as im:
            image = ImageOps.exif_transpose(im).convert('RGB')
        return self.transform(image), row['class_id'] - 1


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(x: torch.Tensor, cfg: dict, size: int) -> torch.Tensor:
    if x.shape[-1] != size:
        x = F.interpolate(x, size=(size, size), mode='bicubic', align_corners=False)
    mean = x.new_tensor(cfg['mean']).view(1, 3, 1, 1)
    std = x.new_tensor(cfg['std']).view(1, 3, 1, 1)
    return (x - mean) / std


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--folds', type=Path, default=Path('artifacts/data/folds.jsonl'))
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--student', type=Path, required=True)
    parser.add_argument('--teacher', type=Path, required=True)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--effective-batch', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--backbone-lr', type=float, default=3e-5)
    parser.add_argument('--head-lr', type=float, default=1e-4)
    parser.add_argument('--temperature', type=float, default=2.)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if args.effective_batch % args.batch:
        parser.error('--effective-batch must be divisible by --batch')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is unavailable')
    args.run.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    student_ckpt = torch.load(args.student, map_location='cpu', weights_only=False)
    teacher_ckpt = torch.load(args.teacher, map_location='cpu', weights_only=False)
    student_cfg, teacher_cfg = student_ckpt['config'], teacher_ckpt['config']
    fold_data = args.folds.read_bytes()
    fold_hash = hashlib.sha256(fold_data).hexdigest()
    if student_cfg['fold'] != teacher_cfg['fold'] or any(
        cfg['fold_sha256'] != fold_hash for cfg in (student_cfg, teacher_cfg)
    ):
        raise ValueError('Teacher and student must have the same frozen training fold')
    if student_ckpt['class_names'] != teacher_ckpt['class_names']:
        raise ValueError('Teacher and student class mappings differ')
    fold = student_cfg['fold']
    rows = [json.loads(line) for line in fold_data.splitlines()]
    train_rows = [row for row in rows if row['fold'] != fold]
    valid_rows = [row for row in rows if row['fold'] == fold]
    config = dict(student_cfg, distillation={
        'teacher_model': teacher_cfg['model'], 'teacher_checkpoint_sha256': file_hash(args.teacher),
        'student_initial_sha256': file_hash(args.student), 'epochs': args.epochs,
        'batch': args.batch, 'effective_batch': args.effective_batch,
        'backbone_lr': args.backbone_lr, 'head_lr': args.head_lr,
        'temperature': args.temperature, 'ce_weight': .5, 'kl_weight': .5, 'seed': args.seed,
    })
    cfg_path = args.run / 'config.json'
    if cfg_path.exists():
        if json.loads(cfg_path.read_text(encoding='utf8')) != json.loads(json.dumps(config)):
            raise ValueError('Run configuration differs; choose a new --run')
    else:
        cfg_path.write_text(json.dumps(config, indent=2), encoding='utf8')
    device = torch.device('cuda')
    student = timm.create_model(student_cfg['model'], pretrained=False, num_classes=102)
    teacher = timm.create_model(teacher_cfg['model'], pretrained=False, num_classes=102)
    student.load_state_dict(student_ckpt['model'])
    teacher.load_state_dict(teacher_ckpt['model'])
    student.to(device)
    teacher.to(device).eval().requires_grad_(False)
    head_ids = {id(param) for param in student.get_classifier().parameters()}
    backbone = [p for p in student.parameters() if id(p) not in head_ids]
    head = [p for p in student.parameters() if id(p) in head_ids]
    optimizer = torch.optim.AdamW([{'params': backbone, 'lr': args.backbone_lr},
                                   {'params': head, 'lr': args.head_lr}], weight_decay=.05)
    scaler = torch.amp.GradScaler('cuda')
    best, start = -1., 0
    last = args.run / 'last.pt'
    if last.exists():
        ckpt = torch.load(last, map_location='cpu', weights_only=False)
        student.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scaler.load_state_dict(ckpt['scaler'])
        restore_rng(ckpt['rng'])
        best, start = ckpt['best_top1'], ckpt['epoch'] + 1
        print(f'Resuming from epoch {start + 1}', flush=True)
    augment_size = max(student_cfg['size'], teacher_cfg['size'])
    train_ds = SharedAugmentFlowers(args.data, train_rows, augment_size, student_cfg['interpolation'])
    valid_ds = Flowers(args.data, valid_rows, student_cfg['size'], False, student_cfg)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch, shuffle=False,
                              num_workers=args.workers, pin_memory=True)
    accum = args.effective_batch // args.batch
    for epoch in range(start, args.epochs):
        decay = .5 * (1 + math.cos(math.pi * epoch / args.epochs))
        optimizer.param_groups[0]['lr'] = args.backbone_lr * decay
        optimizer.param_groups[1]['lr'] = args.head_lr * decay
        generator = torch.Generator().manual_seed(args.seed + epoch)
        loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, generator=generator,
                            num_workers=args.workers, pin_memory=True)
        student.train()
        optimizer.zero_grad(set_to_none=True)
        started, loss_total, seen = time.perf_counter(), 0., 0
        for step, (raw, labels) in enumerate(loader):
            raw, labels = raw.to(device), labels.to(device)
            # The frozen fold has an odd number of training images. MobileNet's
            # BatchNorm head cannot estimate batch statistics from the final
            # single image; use its accumulated statistics for that step.
            if len(labels) == 1:
                for module in student.modules():
                    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                        module.eval()
            student_x = normalize(raw, student_cfg, student_cfg['size'])
            teacher_x = normalize(raw, teacher_cfg, teacher_cfg['size'])
            with torch.no_grad(), torch.autocast(device_type='cuda'):
                teacher_logits = teacher(teacher_x)
            with torch.autocast(device_type='cuda'):
                student_logits = student(student_x)
            ce = F.cross_entropy(student_logits.float(), labels, label_smoothing=.1)
            temperature = args.temperature
            kd = F.kl_div(F.log_softmax(student_logits.float() / temperature, dim=1),
                          F.softmax(teacher_logits.float() / temperature, dim=1),
                          reduction='batchmean')
            loss = .5 * ce + .5 * temperature * temperature * kd
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss at epoch {epoch + 1}, step {step + 1}')
            scaler.scale(loss / accum).backward()
            if (step + 1) % accum == 0 or step + 1 == len(loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            loss_total += float(loss.detach()) * len(labels)
            seen += len(labels)
        metrics = evaluate(student, valid_loader, device, amp=True)
        summary = dict(epoch=epoch + 1, train_loss=loss_total / seen, val_loss=metrics['loss'],
                       top1=metrics['top1'], macro=metrics['macro'], seconds=time.perf_counter() - started,
                       peak_cuda_mb=torch.cuda.max_memory_allocated() / 1e6)
        if metrics['top1'] > best:
            best = metrics['top1']
            weights = {key: value.detach().cpu() for key, value in student.state_dict().items()}
            atomic_save(dict(model=weights, config=config, class_names=student_ckpt['class_names'],
                             epoch=epoch + 1, top1=best), args.run / 'best.pt')
            (args.run / 'best_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf8')
        atomic_save(dict(model=student.state_dict(), optimizer=optimizer.state_dict(),
                         scaler=scaler.state_dict(), rng=rng_state(), epoch=epoch,
                         best_top1=best), last)
        with (args.run / 'history.jsonl').open('a', encoding='utf8') as stream:
            stream.write(json.dumps(summary) + '\n')
        print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
