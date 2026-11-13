#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Export pretrained, batch-one ResNet-18 and an independent PyTorch reference."""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import torch
from torch import fx, nn
from torch.nn.utils.fusion import fuse_conv_bn_eval
from torchvision.models import resnet18, ResNet18_Weights


def fold_batchnorm(model):
    graph = fx.symbolic_trace(copy.deepcopy(model).eval())
    folded = []
    for node in list(graph.graph.nodes):
        if node.op != "call_module":
            continue
        bn = graph.get_submodule(node.target)
        if not isinstance(bn, nn.BatchNorm2d):
            continue
        predecessor = node.args[0]
        if predecessor.op != "call_module" or len(predecessor.users) != 1:
            raise ValueError(f"Cannot fold batch normalization at {node.target}")
        conv = graph.get_submodule(predecessor.target)
        if not isinstance(conv, nn.Conv2d):
            raise ValueError(f"Batch normalization does not follow a convolution: {node.target}")
        parent, _, name = predecessor.target.rpartition(".")
        setattr(graph.get_submodule(parent) if parent else graph, name,
                fuse_conv_bn_eval(conv, bn))
        node.replace_all_uses_with(predecessor)
        graph.graph.erase_node(node)
        folded.append({"conv": predecessor.target, "batchnorm": node.target})
    graph.delete_all_unused_submodules()
    graph.graph.lint()
    graph.recompile()
    return graph, folded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    weights = ResNet18_Weights.IMAGENET1K_V1
    original = resnet18(weights=weights).eval()
    data = torch.randn(1, 3, args.input_size, args.input_size)
    folded, folds = fold_batchnorm(original)
    with torch.no_grad():
        reference = original(data)
        fused_reference = folded(data)
    torch.testing.assert_close(fused_reference, reference, atol=1e-4, rtol=1e-4)
    np.save(args.output / "input.npy", data.numpy())
    np.save(args.output / "reference.npy", reference.numpy())
    np.save(args.output / "folded-reference.npy", fused_reference.numpy())
    (args.output / "torch-fx.txt").write_text(str(folded.graph))
    from torch_mlir import torchscript
    module = torchscript.compile(folded, data, output_type="tosa", use_tracing=True)
    assembly = str(module)
    (args.output / "01_tosa.mlir").write_text(assembly)
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / weights.url.rsplit("/", 1)[-1]
    manifest = {
        "model": "torchvision.resnet18", "weights": str(weights), "weights_url": weights.url,
        "weights_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "input_shape": list(data.shape), "input_kind": "seeded normal tensor", "seed": args.seed,
        "dtype": "float32", "mode": "inference", "batchnorm_folds": folds,
        "folding_max_abs_error": float((reference - fused_reference).abs().max()),
        "tosa_ops": dict(Counter(re.findall(r'\b(tosa\.[a-zA-Z0-9_]+)\b', assembly))),
        "reference_top5": torch.topk(reference[0], 5).indices.tolist(),
    }
    (args.output / "model.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: value for key, value in manifest.items() if key != "batchnorm_folds"}, indent=2))
    print(f"Folded {len(folds)} inference batch normalizations")


if __name__ == "__main__":
    main()
