#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Independent small Conv1D regression: dilation, bias, tails, slopes and two heads."""
import argparse
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/"scripts/reusable_nn"))
from prepare_model import normalize_conv1d


class Fixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.a=nn.Conv1d(2,3,3,stride=2,padding=2,dilation=2)
        self.b=nn.Conv1d(3,5,3,padding=1)
        self.c=nn.Linear(45,4)
        self.d=nn.Linear(45,6)
    def forward(self,x):
        x=F.leaky_relu(self.a(x),0.125)
        x=F.leaky_relu(self.b(x),0.625).flatten(1)
        return self.c(x),self.d(x)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("output",type=Path)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(19)
    model=Fixture().eval()
    x=torch.randn(1,2,17)
    normalized,records=normalize_conv1d(model)
    with torch.no_grad():
        expected=model(x)
        for a,b in zip(normalized(x),expected):
            torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
    np.save(args.output/"input.npy",x.numpy())
    np.save(args.output/"reference.npy",np.concatenate([v.numpy().ravel() for v in expected]))
    from torch_mlir import torchscript
    (args.output/"01_tosa.mlir").write_text(str(torchscript.compile(normalized,x,output_type="tosa",use_tracing=True)))
