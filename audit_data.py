"""Decode every image, fingerprint it, and make repeatable group-stratified folds."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageOps


def dhash(image: Image.Image) -> int:
    gray = ImageOps.grayscale(image).resize((9, 8), Image.Resampling.LANCZOS)
    px = list(gray.get_flattened_data())
    bits = 0
    for y in range(8):
        for x in range(8):
            bits = (bits << 1) | (px[y * 9 + x] > px[y * 9 + x + 1])
    return bits


def make_manifest(root: Path) -> list[dict]:
    rows = []
    for split in ('train', 'valid', 'test'):
        folder = root / split
        classes = sorted((p for p in folder.iterdir() if p.is_dir()), key=lambda p: int(p.name))
        if [int(p.name) for p in classes] != list(range(1, 103)):
            raise ValueError(f'{folder}: expected numeric class folders 1..102')
        for cls in classes:
            for path in sorted(cls.iterdir()):
                if not path.is_file():
                    continue
                raw = path.read_bytes()
                with Image.open(path) as im:
                    im.load()  # complete pixel decode
                    width, height = im.size
                    perceptual = f'{dhash(ImageOps.exif_transpose(im)):016x}'
                rows.append(dict(path=path.relative_to(root).as_posix(), split=split,
                                 class_id=int(cls.name), width=width, height=height,
                                 bytes=len(raw), source='provided_archive',
                                 sha256=hashlib.sha256(raw).hexdigest(), dhash=perceptual))
    return rows


def make_folds(rows: list[dict], confirmed_near: list[list[str]], nfolds: int = 3) -> tuple[list[dict], dict]:
    by_sha = defaultdict(list)
    for row in rows:
        by_sha[row['sha256']].append(row)
    parent = {r['path']: r['path'] for r in rows}

    def find(path):
        while parent[path] != path:
            parent[path] = parent[parent[path]]
            path = parent[path]
        return path

    def union(a, b):
        parent[find(a)] = find(b)

    for group in by_sha.values():
        for row in group[1:]:
            union(group[0]['path'], row['path'])
    for a, b in confirmed_near:
        if a not in parent or b not in parent:
            raise ValueError(f'Unknown confirmed pair: {a}, {b}')
        union(a, b)
    by_group = defaultdict(list)
    for row in rows:
        row['group'] = find(row['path'])
        row.setdefault('source', 'provided_archive')
        by_group[row['group']].append(row)
    conflicts = []
    leakage = []
    for group in by_group.values():
        classes = {r['class_id'] for r in group}
        if len(classes) > 1:
            conflicts.append([r['path'] for r in group])
        if {'test'} & {r['split'] for r in group} and len({r['split'] for r in group}) > 1:
            leakage.append([r['path'] for r in group])
    dev = [r for r in rows if r['split'] != 'test']
    banned = {path for group in conflicts + leakage for path in group}
    dev = [r for r in dev if r['path'] not in banned]
    # Keep one representative of each byte-identical development image.
    unique_dev = {}
    excluded_exact = []
    for row in sorted(dev, key=lambda r: r['path']):
        if row['sha256'] in unique_dev:
            excluded_exact.append(row['path'])
        else:
            unique_dev[row['sha256']] = row
    dev = list(unique_dev.values())
    grouped = defaultdict(list)
    for row in dev:
        grouped[(row['class_id'], find(row['path']))].append(row)
    fold_rows = []
    for class_id in range(1, 103):
        groups = sorted(((group_id, items) for (cls, group_id), items in grouped.items() if cls == class_id),
                        key=lambda x: x[0])
        loads = [0] * nfolds
        for group_id, items in sorted(groups, key=lambda x: (-len(x[1]), x[0])):
            fold = min(range(nfolds), key=lambda f: (loads[f], f))
            loads[fold] += len(items)
            fold_rows.extend(dict(path=r['path'], class_id=class_id, fold=fold,
                                  group=group_id, sha256=r['sha256']) for r in items)
    report = dict(total=len(rows), splits=dict(Counter(r['split'] for r in rows)),
                  classes=len({r['class_id'] for r in rows}),
                  exact_duplicate_groups=sum(len(g) > 1 for g in by_sha.values()),
                  confirmed_near_pairs=len(confirmed_near),
                  cross_label_groups=conflicts, test_dev_duplicate_groups=leakage,
                  excluded_exact_duplicate_paths=excluded_exact,
                  excluded_dev_paths=sorted((banned & {r['path'] for r in rows if r['split'] != 'test'}) | set(excluded_exact)),
                  fold_counts=dict(Counter(r['fold'] for r in fold_rows)))
    return fold_rows, report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, default=Path('artifacts/data'))
    p.add_argument('--confirmed-near', type=Path, default=Path('confirmed_near.json'))
    p.add_argument('--reuse-manifest', action='store_true', help='Use a previously decoded manifest without rechecking files')
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / 'manifest.jsonl'
    if args.reuse_manifest and manifest_path.exists():
        rows = [json.loads(line) for line in manifest_path.read_text(encoding='utf8').splitlines()]
    else:
        rows = make_manifest(args.data)
    confirmed_near = json.loads(args.confirmed_near.read_text(encoding='utf8'))
    folds, report = make_folds(rows, confirmed_near)
    for name, data in [('manifest.jsonl', rows), ('folds.jsonl', folds)]:
        with (args.out / name).open('w', encoding='utf8') as f:
            for row in data:
                f.write(json.dumps(row, sort_keys=True) + '\n')
    (args.out / 'audit.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf8')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
