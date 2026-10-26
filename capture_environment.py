"""Record the actual machine and library versions used for experiments."""
import json
import platform
import subprocess
import sys
from pathlib import Path

import PIL
import timm
import torch
import torchvision


query = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'],
                       capture_output=True, text=True, check=True)
gpu = query.stdout.strip()
info = dict(python=sys.version.split()[0], platform=platform.platform(), processor=platform.processor(),
            torch=torch.__version__, torchvision=torchvision.__version__, timm=timm.__version__,
            pillow=PIL.__version__, cuda_runtime=torch.version.cuda,
            cuda_available=torch.cuda.is_available(), gpu_nvidia_smi=gpu,
            gpu_capability=torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
            cpu_threads=torch.get_num_threads())
Path('artifacts/environment.json').write_text(json.dumps(info, indent=2), encoding='utf8')
print(json.dumps(info, indent=2))
