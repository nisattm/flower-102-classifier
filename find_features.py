"""Find same-photo candidates using pretrained EfficientNet image embeddings."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import timm
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from train import Flowers, MODELS


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--manifest', type=Path, default=Path('artifacts/data/manifest.jsonl'))
    p.add_argument('--output', type=Path, default=Path('artifacts/data/feature_candidates.json'))
    p.add_argument('--threshold', type=float, default=.985)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--reuse-features', action='store_true')
    args = p.parse_args()
    rows = [json.loads(line) for line in args.manifest.read_text(encoding='utf8').splitlines()]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feature_path = args.output.with_name('features.pt')
    if args.reuse_features:
        payload = torch.load(feature_path, map_location='cpu', weights_only=True)
        if payload['paths'] != [r['path'] for r in rows]:
            raise ValueError('Cached features do not match manifest paths')
        features = payload['features'].float()
    else:
        model = timm.create_model(MODELS['efficient'], pretrained=True, num_classes=0).to(device).eval()
        config = timm.data.resolve_model_data_config(model)
        dataset = Flowers(args.data, rows, 224, False, config)
        loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=0)
        features = []
        with torch.inference_mode():
            for step, (x, _) in enumerate(loader):
                feature = F.normalize(model(x.to(device)).float(), dim=1)
                features.append(feature.cpu())
                if (step + 1) % 100 == 0:
                    print(f'Embedded {min((step + 1) * args.batch, len(rows))}/{len(rows)}', flush=True)
        features = torch.cat(features)
        torch.save(dict(paths=[r['path'] for r in rows], features=features.half()), feature_path)
    known = {tuple(sorted(pair)) for pair in json.loads(Path('confirmed_near.json').read_text())}
    matches = {}
    matrix = features.to(device)
    for start in range(0, len(rows), 512):
        scores = matrix[start:start + 512] @ matrix.T
        for offset in range(scores.shape[0]):
            scores[offset, start + offset] = -1
        vals, inds = scores.topk(12, dim=1)
        for offset in range(scores.shape[0]):
            i = start + offset
            for value, j in zip(vals[offset].tolist(), inds[offset].tolist()):
                if value < args.threshold or rows[i]['sha256'] == rows[j]['sha256']:
                    continue
                paths = tuple(sorted((rows[i]['path'], rows[j]['path'])))
                if paths in known:
                    continue
                matches[paths] = max(value, matches.get(paths, -1))
    candidates = [dict(left=a, right=b, similarity=score,
                       cross_split=a.split('/')[0] != b.split('/')[0],
                       same_class=a.split('/')[1] == b.split('/')[1])
                  for (a, b), score in sorted(matches.items(), key=lambda item: -item[1])]
    args.output.write_text(json.dumps(candidates, indent=2), encoding='utf8')
    print(json.dumps(dict(candidates=len(candidates),
                          cross_split=sum(c['cross_split'] for c in candidates),
                          cross_class=sum(not c['same_class'] for c in candidates)), indent=2))


if __name__ == '__main__':
    main()
