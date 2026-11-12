# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Semantics-preserving frontend normalization for the reusable TOSA backend."""
import copy
from torch import fx, nn


def normalize_conv1d(model):
    """Extract native Conv1d parameters and express unsupported export ops as Conv2d.

    Leaves the original model untouched. Groups/nonzero padding modes are rejected
    because the reusable convolution contract currently implements dense convolution.
    """
    graph = fx.symbolic_trace(copy.deepcopy(model).eval())
    records = []
    sources = {node.target:graph.get_submodule(node.target) for node in graph.graph.nodes
               if node.op == "call_module"}
    for node in list(graph.graph.nodes):
        if node.op != "call_module":
            continue
        source = sources[node.target]
        if not isinstance(source, nn.Conv1d):
            continue
        if source.groups != 1 or source.padding_mode != "zeros" or isinstance(source.padding, str):
            raise ValueError("Conv1D normalization requires dense convolution with explicit zero padding")
        target = nn.Conv2d(source.in_channels, source.out_channels, (1,source.kernel_size[0]),
                           stride=(1,source.stride[0]), padding=(0,source.padding[0]),
                           dilation=(1,source.dilation[0]), bias=source.bias is not None,
                           device=source.weight.device, dtype=source.weight.dtype)
        target.weight = nn.Parameter(source.weight.detach().unsqueeze(2).clone())
        if source.bias is not None:
            target.bias = nn.Parameter(source.bias.detach().clone())
        parent, _, name = node.target.rpartition(".")
        setattr(graph.get_submodule(parent) if parent else graph, name, target)
        with graph.graph.inserting_before(node):
            expanded = graph.graph.call_method("unsqueeze", (node.args[0],2))
        node.args = (expanded,)
        with graph.graph.inserting_after(node):
            squeezed = graph.graph.call_method("squeeze", (node,2))
        node.replace_all_uses_with(squeezed)
        squeezed.args = (node,2)
        records.append({"module":node.target, "operator":"conv1d", "in_channels":source.in_channels,
                        "out_channels":source.out_channels, "kernel":source.kernel_size[0],
                        "stride":source.stride[0], "padding":source.padding[0],
                        "dilation":source.dilation[0], "groups":source.groups})
    graph.graph.lint()
    graph.recompile()
    return graph, records
