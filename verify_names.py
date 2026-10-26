"""Validate community class-ID mapping against Oxford's category thumbnails and counts."""
from __future__ import annotations

import html
import json
import os
import re
from collections import Counter
from pathlib import Path


def main() -> None:
    names_path = Path('class_names.json')
    names = json.loads(names_path.read_text(encoding='utf8'))
    assert set(names) == {str(i) for i in range(1, 103)}
    if names['78'] == 'lotus lotus':
        names['78'] = 'lotus'
        names_path.write_text(json.dumps(names, indent=2, ensure_ascii=False, sort_keys=True) + '\n', encoding='utf8')
    source = Path('artifacts/data/oxford_categories.html').read_text(encoding='utf8')
    pattern = re.compile(r'<img src="thumbs/thumbim_(\d+)\.jpg"[^>]*></td><td class=lab[LC R]*>(.*?)</td><td class=num[LC R]*>(\d+)</td>')
    found = [(f'image_{image_id}.jpg', html.unescape(name).strip().rstrip('?').strip(), int(count))
             for image_id, name, count in pattern.findall(source)]
    by_name = {name: (image, count) for image, name, count in found}
    root = Path(os.environ.get('FLOWER_DATA_ROOT', 'flower_data'))
    manifest = [json.loads(line) for line in Path('artifacts/data/manifest.jsonl').read_text().splitlines()]
    count_by_class = Counter(row['class_id'] for row in manifest)
    mismatches = []
    for key, name in names.items():
        if name not in by_name:
            mismatches.append([key, name, 'name missing from Oxford page'])
            continue
        image, expected_count = by_name[name]
        class_id = int(key)
        actual_image = next((row['path'] for row in manifest if row['class_id'] == class_id and row['path'].endswith(image)), None)
        if not actual_image:
            mismatches.append([key, name, f'thumbnail image {image} absent'])
        if count_by_class[class_id] != expected_count:
            mismatches.append([key, name, f'count {count_by_class[class_id]} != {expected_count}'])
    report = dict(oxford_categories=len(found), mapping_categories=len(names), mismatches=mismatches)
    Path('artifacts/data/name_verification.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
