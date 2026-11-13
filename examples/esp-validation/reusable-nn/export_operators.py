#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Export independent reuse tests, each containing only one accelerator type."""
import argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch_mlir import torchscript


class Convolution(nn.Module):
    def __init__(self, channels=(9, 11)):
        super().__init__()
        first, second = channels
        self.stem = nn.Conv2d(3, first, 7, stride=2, padding=3)
        self.dilated = nn.Conv2d(first, second, 3, padding=2, dilation=2)
        self.pointwise = nn.Conv2d(second, 5, 1)

    def forward(self, x):
        return self.pointwise(torch.relu(self.dilated(torch.relu(self.stem(x)))))


class Addition(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("a", torch.arange(143).reshape(1, 1, 11, 13).float()/32-2)
        self.register_buffer("b", torch.arange(143).reshape(1, 1, 13, 11).float()/16-4)

    def forward(self, x):
        y = torch.relu(x + self.a)
        return y.reshape(1, 1, 13, 11) + self.b


class Gemm(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.empty(5, 7))
        self.b = nn.Parameter(torch.empty(7, 2))

    def forward(self, x):
        return torch.matmul(torch.matmul(x, self.a), self.b)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=Path)
parser.add_argument("--wide-channels", action="store_true",
                    help="Exercise multiple 16-channel reduction tiles and channel tails")
args = parser.parse_args()
torch.set_num_threads(2)
channels = (19, 23) if args.wide_channels else (9, 11)
for kind, model, shape in (("conv", Convolution(channels), (1, 3, 5, 7)),
                           ("add", Addition(), (1, 1, 11, 13)),
                           ("gemm", Gemm(), (1, 3, 5))):
    model.eval()
    with torch.no_grad():
        for i, p in enumerate(model.parameters()):
            p.copy_(((torch.arange(p.numel()).reshape(p.shape)*7+i*3)%17-8)/32)
    data = ((torch.arange(np.prod(shape)).reshape(shape)*5)%19-9).float()/16
    with torch.no_grad():
        output = model(data)
    folder = args.output/kind
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder/"input.npy", data.numpy())
    np.save(folder/"reference.npy", output.numpy())
    module = torchscript.compile(model, data, output_type="tosa", use_tracing=True)
    (folder/"01_tosa.mlir").write_text(str(module))
    print(kind, list(shape), "->", list(output.shape))
