"""Download and inspect pretrained baselines without using the GPU."""
import json
import sys

import timm

from train import MODELS


for key in sys.argv[1:] or ('mobile', 'teacher'):
    model = timm.create_model(MODELS[key], pretrained=True, num_classes=102)
    params = sum(p.numel() for p in model.parameters())
    print(json.dumps(dict(model=key, name=MODELS[key], parameters=params,
                          data_config=timm.data.resolve_model_data_config(model))), flush=True)
