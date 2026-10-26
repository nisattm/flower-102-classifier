"""Check that held-out test and duplicate groups cannot leak into development folds."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    root = Path('artifacts/data')
    manifest = [json.loads(line) for line in (root / 'manifest.jsonl').read_text().splitlines()]
    folds = [json.loads(line) for line in (root / 'folds.jsonl').read_text().splitlines()]
    audit = json.loads((root / 'audit.json').read_text())
    test = {row['path'] for row in manifest if row['split'] == 'test'}
    actual = {row['path'] for row in folds}
    assert not actual & test
    assert len(actual) == len(folds)
    assert len(actual) + len(audit['excluded_dev_paths']) + len(test) == len(manifest)
    groups = defaultdict(set)
    for row in folds:
        groups[row['group']].add(row['fold'])
    assert all(len(value) == 1 for value in groups.values())
    for fold in range(3):
        counts = Counter(row['class_id'] for row in folds if row['fold'] == fold)
        assert set(counts) == set(range(1, 103))
        assert min(counts.values()) > 0
    print(json.dumps(dict(dev=len(folds), held_out_test=len(test), excluded=len(audit['excluded_dev_paths']),
                          groups=len(groups), status='passed'), indent=2))


if __name__ == '__main__':
    main()
