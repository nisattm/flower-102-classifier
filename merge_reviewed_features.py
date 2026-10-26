"""Merge the visually reviewed >=0.95 feature candidate list into group evidence."""
import json
from pathlib import Path

target = Path('confirmed_near.json')
confirmed = json.loads(target.read_text(encoding='utf8'))
candidates = json.loads(Path('artifacts/data/feature_candidates_95.json').read_text(encoding='utf8'))
known = {tuple(sorted(pair)) for pair in confirmed}
for item in candidates:
    pair = tuple(sorted((item['left'], item['right'])))
    if pair not in known:
        confirmed.append(list(pair))
        known.add(pair)
target.write_text(json.dumps(confirmed, indent=2) + '\n', encoding='utf8')
print(f'{len(candidates)} reviewed feature pairs merged; {len(confirmed)} total pairs')
