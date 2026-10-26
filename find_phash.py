"""Search another perceptual hash for near-duplicate candidates."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


N = 32
COS = np.array([[np.cos(np.pi * (2 * x + 1) * k / (2 * N)) for x in range(N)]
                for k in range(N)], dtype=np.float32)


def phash(path: Path) -> int:
    with Image.open(path) as im:
        pixels = np.asarray(ImageOps.grayscale(ImageOps.exif_transpose(im)).resize((N, N), Image.Resampling.LANCZOS),
                            dtype=np.float32)
    dct = COS @ pixels @ COS.T
    low = dct[:8, :8].copy()
    values = low.ravel()
    threshold = np.median(values[1:])
    bits = 0
    for value in values:
        bits = (bits << 1) | int(value > threshold)
    return bits


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--manifest', type=Path, default=Path('artifacts/data/manifest.jsonl'))
    p.add_argument('--out', type=Path, default=Path('artifacts/data/phash_candidates.json'))
    p.add_argument('--distance', type=int, default=6)
    args = p.parse_args()
    rows = [json.loads(line) for line in args.manifest.read_text(encoding='utf8').splitlines()]
    hashes = [phash(args.data / row['path']) for row in rows]
    with args.out.with_name('phash.jsonl').open('w', encoding='utf8') as f:
        for row, value in zip(rows, hashes):
            f.write(json.dumps({'path': row['path'], 'phash': f'{value:016x}'}) + '\n')
    # Seven bit ranges guarantee one identical range for distance <=6.
    ranges = [(0, 10), (10, 19), (19, 28), (28, 37), (37, 46), (46, 55), (55, 64)]
    buckets = defaultdict(list)
    for i, (row, value) in enumerate(zip(rows, hashes)):
        for part, (lo, hi) in enumerate(ranges):
            buckets[(row['class_id'], part, (value >> lo) & ((1 << (hi - lo)) - 1))].append(i)
    pairs = set()
    for items in buckets.values():
        for pos, a in enumerate(items):
            for b in items[pos + 1:]:
                pairs.add((a, b))
    known = {tuple(sorted(pair)) for pair in json.loads(Path('confirmed_near.json').read_text())}
    matches = []
    for a, b in sorted(pairs):
        left, right = rows[a], rows[b]
        paths = tuple(sorted((left['path'], right['path'])))
        if left['sha256'] == right['sha256'] or paths in known:
            continue
        distance = (hashes[a] ^ hashes[b]).bit_count()
        if distance <= args.distance:
            matches.append(dict(left=paths[0], right=paths[1], distance=distance,
                                cross_split=left['split'] != right['split']))
    args.out.write_text(json.dumps(matches, indent=2), encoding='utf8')
    print(json.dumps(dict(candidates=len(matches), cross_split=sum(m['cross_split'] for m in matches)), indent=2))


if __name__ == '__main__':
    main()
