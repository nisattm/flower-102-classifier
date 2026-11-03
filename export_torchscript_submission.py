"""Export the frozen Flower 102 checkpoint to the instructor's one-file format.

The evaluator supplies RGB float tensors in [0, 1] after resizing and cropping.
The TorchScript graph therefore contains the training mean/std normalization,
the network, and softmax. Output column i represents class i + 1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import timm
import torch
from torch import nn


class NormalizedProbabilities(nn.Module):
    def __init__(self, network: nn.Module, mean: tuple[float, ...], std: tuple[float, ...]):
        super().__init__()
        self.network = network
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        normalized = (images - self.mean) / self.std
        return torch.softmax(self.network(normalized), dim=1)


def check_output(output: torch.Tensor, batch: int) -> None:
    if not isinstance(output, torch.Tensor) or output.shape != (batch, 102):
        raise ValueError(f"Expected [{batch}, 102] tensor, got {type(output)} {getattr(output, 'shape', None)}")
    if not torch.isfinite(output).all() or output.min() < 0 or output.max() > 1:
        raise ValueError("Output is not a finite probability tensor in [0, 1]")
    if not torch.allclose(output.sum(dim=1), torch.ones(batch), atol=1e-5, rtol=0):
        raise ValueError("Probability rows do not sum to one")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("submission/model.pth"))
    parser.add_argument("--output", type=Path, default=Path("submission_ready/model.pt"))
    args = parser.parse_args()

    checkpoint = torch.load(args.source, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if config["size"] != 224:
        raise ValueError(f"Expected 224-pixel training input, got {config['size']}")
    if set(checkpoint["class_names"]) != {str(i) for i in range(1, 103)}:
        raise ValueError("The source checkpoint does not contain all 102 class IDs")
    mean = tuple(float(x) for x in config["mean"])
    std = tuple(float(x) for x in config["std"])
    if len(mean) != 3 or len(std) != 3 or any(x <= 0 for x in std):
        raise ValueError("Invalid saved mean/std normalization")

    network = timm.create_model(config["model"], pretrained=False, num_classes=102)
    network.load_state_dict(checkpoint["model"], strict=True)
    wrapped = NormalizedProbabilities(network.float().eval().cpu(), mean, std).eval()

    torch.manual_seed(42)
    with torch.inference_mode():
        traced = torch.jit.trace(wrapped, torch.rand(1, 3, 224, 224), strict=True)
        for batch in (1, 2, 8):
            example = torch.rand(batch, 3, 224, 224)
            eager_output = wrapped(example)
            traced_output = traced(example)
            check_output(traced_output, batch)
            if not torch.allclose(traced_output, eager_output, atol=1e-5, rtol=1e-4):
                raise ValueError(f"Trace differs from the trained network at batch size {batch}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    traced.save(str(temporary))
    reloaded = torch.jit.load(str(temporary), map_location="cpu").eval()
    with torch.inference_mode():
        check_output(reloaded(torch.rand(2, 3, 224, 224)), 2)
    temporary.replace(args.output)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(json.dumps({
        "file": str(args.output),
        "bytes": args.output.stat().st_size,
        "sha256": digest,
        "model": config["model"],
        "normalization_mean": mean,
        "normalization_std": std,
        "class_order": "output column i is class i + 1",
        "checked_batches": [1, 2, 8],
    }, indent=2))


if __name__ == "__main__":
    main()
