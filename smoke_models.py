"""Check pretrained candidate forward passes and normalization on the local GPU."""
import json
import os
from pathlib import Path

import timm
import torch
from torch.utils.data import DataLoader

from train import Flowers, MODELS

rows = [json.loads(line) for line in Path('artifacts/data/folds.jsonl').read_text().splitlines()]
rows = [row for row in rows if row['fold'] != 0][:8]
for key, batch in [('mobile', 8), ('teacher', 2)]:
    model = timm.create_model(MODELS[key], pretrained=True, num_classes=102).cuda().eval()
    config = timm.data.resolve_model_data_config(model)
    data = Flowers(Path(os.environ.get('FLOWER_DATA_ROOT', 'flower_data')), rows, 224, True, config)
    x, y = next(iter(DataLoader(data, batch_size=batch)))
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode(), torch.autocast('cuda'):
        logits = model(x.cuda())
        loss = torch.nn.functional.cross_entropy(logits.float(), y.cuda())
    print(json.dumps(dict(model=key, shape=list(logits.shape), finite=bool(torch.isfinite(loss)),
                          loss=float(loss), peak_cuda_mb=torch.cuda.max_memory_allocated() / 1e6)), flush=True)
    del model
    torch.cuda.empty_cache()
