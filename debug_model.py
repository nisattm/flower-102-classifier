from pathlib import Path
import os
import json

import timm
import torch
from torch.utils.data import DataLoader

from train import Flowers, MODELS

rows = [json.loads(line) for line in Path('artifacts/data/folds.jsonl').read_text().splitlines()]
rows = [r for r in rows if r['fold'] != 0][:16]
model = timm.create_model(MODELS['efficient'], pretrained=True, num_classes=102).cuda().eval()
data = Flowers(Path(os.environ.get('FLOWER_DATA_ROOT', 'flower_data')), rows, 224, True,
               timm.data.resolve_model_data_config(model))
x, y = next(iter(DataLoader(data, batch_size=8)))
print('input', x.shape, torch.isfinite(x).all().item(), x.min().item(), x.max().item(), flush=True)
x, y = x.cuda(), y.cuda()
with torch.inference_mode():
    cpu_output = model.cpu()(x.cpu())
print('cpu', torch.isfinite(cpu_output).all().item(), cpu_output.min().item(), cpu_output.max().item(), flush=True)
model.cuda()
for amp in (False, True):
    with torch.autocast('cuda', enabled=amp):
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.float(), y, label_smoothing=.1)
    print('amp', amp, 'output', torch.isfinite(logits).all().item(), logits.min().item(), logits.max().item(),
          'loss', loss.item(), flush=True)
