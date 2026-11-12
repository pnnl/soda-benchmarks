#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Export the local 240-bin EELS inference encoder with its actual reference weights."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.layers = nn.ModuleList()
        for i, (ci, co, k) in enumerate(((1,16,7),(16,32,7),(32,32,3),(32,64,3)), 1):
            layer = nn.Conv1d(ci, co, k, stride=2)
            w = np.fromfile(source / f"models/l{i}_w.bin", dtype="<f4").reshape(k,ci,co)
            b = np.fromfile(source / f"models/l{i}_b.bin", dtype="<f4")
            with torch.no_grad():
                layer.weight.copy_(torch.from_numpy(w.transpose(2,1,0).copy()))
                layer.bias.copy_(torch.from_numpy(b))
            self.layers.append(layer)
        self.heads = nn.ModuleList()
        for name in ("zm", "zv"):
            layer = nn.Linear(960,16)
            w = np.fromfile(source / f"models/{name}_w.bin", dtype="<f4").reshape(960,16)
            b = np.fromfile(source / f"models/{name}_b.bin", dtype="<f4")
            with torch.no_grad():
                layer.weight.copy_(torch.from_numpy(w.T.copy()))
                layer.bias.copy_(torch.from_numpy(b))
            self.heads.append(layer)

    def forward(self, x):
        for layer in self.layers:
            # TensorFlow SAME for these even lengths, stride two, odd kernels.
            total = layer.kernel_size[0] - 2
            x = F.leaky_relu(layer(F.pad(x, (total//2, total-total//2))), 0.3)
        x = x.transpose(1,2).flatten(1)  # preserve the reference's length-major order
        return self.heads[0](x), self.heads[1](x)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    model = Encoder(args.source).eval()
    data = np.fromfile(args.source / "data/input.bin", dtype="<f4").reshape(1,1,240)
    with torch.no_grad():
        results = model(torch.from_numpy(data))
    reference = np.concatenate([x.numpy().ravel() for x in results])
    golden = np.concatenate([np.fromfile(args.source / f"data/golden_{name}.bin", dtype="<f4")
                             for name in ("z_mean", "z_log_var")])
    np.testing.assert_allclose(reference, golden, atol=1e-6, rtol=1e-5)
    np.save(args.output / "input.npy", data)
    np.save(args.output / "reference.npy", golden)
    np.save(args.output / "pytorch-reference.npy", reference)
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts/reusable_nn"))
    from prepare_model import normalize_conv1d
    normalized, conv1d = normalize_conv1d(model)
    with torch.no_grad():
        for actual, expected in zip(normalized(torch.from_numpy(data)), results):
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    from torch_mlir import torchscript
    module = torchscript.compile(normalized, torch.from_numpy(data), output_type="tosa", use_tracing=True)
    (args.output / "01_tosa.mlir").write_text(str(module))
    paths = sorted((args.source / "models").glob("*.bin")) + sorted((args.source / "data").glob("*.bin"))
    manifest = {"model":"EELS encoder", "input_shape":list(data.shape), "dtype":"float32",
                "outputs":[{"name":"z_mean","shape":[1,16]},{"name":"z_log_var","shape":[1,16]}],
                "conv1d_normalization":conv1d,
                "pytorch_vs_golden_max_abs_error":float(np.max(np.abs(reference-golden))),
                "weight_training_provenance":"not established; supplied local reference fixture",
                "source_sha256":{str(f.relative_to(args.source)):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths}}
    (args.output / "model.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({k:v for k,v in manifest.items() if k!="source_sha256"}, indent=2))


if __name__ == "__main__":
    main()
