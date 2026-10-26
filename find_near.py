"""Find candidate near duplicates for manual review; never removes images automatically."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, default=Path('artifacts/data/manifest.jsonl'))
    p.add_argument('--output', type=Path, default=Path('artifacts/data/near_candidates.json'))
    p.add_argument('--distance', type=int, default=4)
    args = p.parse_args()
    rows = [json.loads(line) for line in args.manifest.read_text(encoding='utf8').splitlines()]
    # Five bit ranges ensure that Hamming distance <=4 shares at least one range.
    ranges = [(0, 13), (13, 26), (26, 39), (39, 52), (52, 64)]
    buckets = defaultdict(list)
    for i, row in enumerate(rows):
        value = int(row['dhash'], 16)
        for part, (lo, hi) in enumerate(ranges):
            buckets[(row['class_id'], part, (value >> lo) & ((1 << (hi - lo)) - 1))].append(i)
    pairs = set()
    for items in buckets.values():
        for a_pos, a in enumerate(items):
            for b in items[a_pos + 1:]:
                pairs.add((a, b))
    matches = []
    for a, b in sorted(pairs):
        left, right = rows[a], rows[b]
        if left['sha256'] == right['sha256']:
            continue
        distance = (int(left['dhash'], 16) ^ int(right['dhash'], 16)).bit_count()
        if distance <= args.distance:
            matches.append(dict(left=left['path'], right=right['path'], distance=distance,
                                cross_split=left['split'] != right['split']))
    args.output.write_text(json.dumps(matches, indent=2), encoding='utf8')
    print(json.dumps(dict(candidates=len(matches), cross_split=sum(m['cross_split'] for m in matches),
                          output=str(args.output)), indent=2))


if __name__ == '__main__':
    main()
