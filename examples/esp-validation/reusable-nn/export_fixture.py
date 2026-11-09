#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Small residual graph covering the same operator contracts as ResNet-18."""
import argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch_mlir import torchscript

class ResidualFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem=nn.Conv2d(3,9,7,stride=2,padding=3)
        self.first=nn.Conv2d(9,11,3,padding=1)
        self.second=nn.Conv2d(11,11,3,padding=1)
        self.projection=nn.Conv2d(9,11,1)
        self.fc=nn.Linear(11,7)
    def forward(self,x):
        x=torch.relu(self.stem(x))
        x=torch.nn.functional.max_pool2d(x,3,2,1)
        y=torch.relu(self.first(x))
        y=torch.relu(self.second(y)+self.projection(x))
        y=torch.nn.functional.adaptive_avg_pool2d(y,(1,1))
        return self.fc(torch.flatten(y,1))

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("output",type=Path)
args=parser.parse_args()
args.output.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(2)
model=ResidualFixture().eval()
with torch.no_grad():
    for i,p in enumerate(model.parameters()):
        values=((torch.arange(p.numel()).reshape(p.shape)*7+i*3)%17-8)/32
        p.copy_(values)
data=((torch.arange(3*7*9).reshape(1,3,7,9)*5)%19-9).float()/16
with torch.no_grad(): output=model(data)
np.save(args.output/"input.npy",data.numpy())
np.save(args.output/"reference.npy",output.numpy())
module=torchscript.compile(model,data,output_type="tosa",use_tracing=True)
(args.output/"01_tosa.mlir").write_text(str(module))
print("Exported small residual graph",list(data.shape),"->",list(output.shape))
