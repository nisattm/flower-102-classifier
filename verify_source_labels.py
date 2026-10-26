"""Compare every provided folder label to Oxford's original imagelabels.mat."""
from __future__ import annotations

import json
import re
import struct
import zlib
from pathlib import Path


def main() -> None:
    raw = Path('artifacts/data/imagelabels.mat').read_bytes()
    if not raw.startswith(b'MATLAB 5.0 MAT-file') or struct.unpack_from('<I', raw, 128)[0] != 15:
        raise ValueError('Unexpected Oxford MATLAB file format')
    compressed_size = struct.unpack_from('<I', raw, 132)[0]
    matrix = zlib.decompress(raw[136:136 + compressed_size])
    # This official file stores a 1 x 8,189 uint8 array named "labels".
    dtype, size = struct.unpack_from('<II', matrix, 56)
    if matrix[48:54] != b'labels' or dtype != 2 or size != 8189:
        raise ValueError('Unexpected labels array layout')
    labels = matrix[64:64 + size]
    rows = [json.loads(line) for line in Path('artifacts/data/manifest.jsonl').read_text(encoding='utf8').splitlines()]
    mismatches = []
    seen_ids = set()
    for row in rows:
        image_id = int(re.search(r'image_(\d+)\.jpg$', row['path']).group(1))
        seen_ids.add(image_id)
        official = int(labels[image_id - 1])
        if official != row['class_id']:
            mismatches.append(dict(path=row['path'], provided=row['class_id'], official=official))
    report = dict(official_count=len(labels), archive_count=len(rows), unique_image_ids=len(seen_ids),
                  complete_ids=seen_ids == set(range(1, len(labels) + 1)), mismatches=mismatches)
    Path('artifacts/data/source_label_verification.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
