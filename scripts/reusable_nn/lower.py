#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Lower a checked TOSA graph to reusable operator calls, independently of tiles.

Uses MLIR's parser/types/attributes, not a model-specific list of layers. Constants
are externalized to a parameter tensor. NHWC propagation, pointwise fusion and
arena liveness operate on the typed data-dependency graph.
"""
import argparse
from collections import Counter
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from mlir import ir
from mlir.dialects import arith, func, memref, tosa


@dataclass(eq=False)
class Node:
    kind: str
    shape: tuple
    inputs: tuple = ()
    attrs: dict = field(default_factory=dict)
    data: object = None
    offset: int = -1
    name: str = ""

    @property
    def elements(self):
        return math.prod(self.shape)


def ordered(root):
    result, seen = [], set()
    def visit(node):
        if node in seen:
            return
        seen.add(node)
        for parent in node.inputs:
            visit(parent)
        result.append(node)
    visit(root)
    return result


def transpose(x, permutation):
    p = tuple(permutation)
    if sorted(p) != list(range(len(x.shape))):
        raise ValueError("Invalid transpose permutation")
    if x.kind == "transpose":
        return transpose(x.inputs[0], tuple(x.attrs["perm"][i] for i in p))
    if p == tuple(range(len(p))):
        return x
    if x.kind == "const":
        data = np.transpose(x.data, p)
        return Node("const", data.shape, data=data)
    return Node("transpose", tuple(x.shape[i] for i in p), (x,), {"perm": p})


def pointwise(kind, parents, attrs):
    # Pointwise operations commute with an identical permutation on every input.
    if all(x.kind == "transpose" for x in parents):
        p = parents[0].attrs["perm"]
        if all(x.attrs["perm"] == p for x in parents):
            return transpose(pointwise(kind, tuple(x.inputs[0] for x in parents), attrs), p)
    if any(x.shape != parents[0].shape for x in parents):
        raise ValueError("Reusable pointwise operations currently require equal shapes")
    return Node(kind, parents[0].shape, parents, attrs)


def parse_graph(text):
    module = ir.Module.parse(text)
    module.operation.verify()
    functions = [op for op in module.body.operations if op.operation.name == "func.func"]
    if len(functions) != 1 or len(functions[0].regions[0].blocks) != 1:
        raise ValueError("Expected one single-block function")
    block = functions[0].regions[0].blocks[0]
    if len(block.arguments) != 1:
        raise ValueError("Initial network lowering requires one input tensor")
    def shape(value):
        t = ir.RankedTensorType(value.type)
        if not t.has_static_shape or not 1 <= t.rank <= 4 or any(d <= 0 for d in t.shape):
            raise ValueError("Expected positive static tensors of rank one to four")
        if not ir.F32Type.isinstance(t.element_type):
            raise ValueError("Network arithmetic must use float32")
        return tuple(t.shape)
    inputs = Node("input", shape(block.arguments[0]))
    values = {block.arguments[0]: inputs}
    root = None
    for op in block.operations:
        name = op.operation.name
        if name == "func.return":
            if len(op.operands) != 1:
                raise ValueError("Expected one output tensor")
            root = values[op.operands[0]]
            continue
        if name == "tosa.const":
            data = np.array(ir.DenseElementsAttr(op.attributes["value"]), copy=True)
            values[op.results[0]] = Node("const", data.shape, data=data)
            continue
        target = shape(op.results[0])
        parents = tuple(values[v] for v in op.operands)
        attrs = {}
        for attr in op.attributes:
            if ir.DenseI64ArrayAttr.isinstance(attr.attr):
                attrs[attr.name] = tuple(ir.DenseI64ArrayAttr(attr.attr))
        if name == "tosa.transpose":
            if parents[1].kind != "const":
                raise ValueError("Transpose permutation must be constant")
            node = transpose(parents[0], parents[1].data.tolist())
        elif name == "tosa.reshape":
            if parents[0].elements != math.prod(target):
                raise ValueError("Reshape changes the number of elements")
            if parents[0].kind == "const":
                node = Node("const", target, data=parents[0].data.reshape(target))
            else:
                node = Node("reshape", target, parents)
        elif name == "tosa.clamp":
            lo = ir.FloatAttr(op.attributes["min_fp"]).value
            hi = ir.FloatAttr(op.attributes["max_fp"]).value
            if lo != 0 or hi != float(np.finfo(np.float32).max):
                raise ValueError("Only the float32 ReLU clamp is supported")
            node = pointwise("relu", parents, {})
        elif name == "tosa.add":
            node = pointwise("add", parents, {})
        elif name in ("tosa.conv2d", "tosa.matmul", "tosa.max_pool2d", "tosa.avg_pool2d"):
            node = Node(name.removeprefix("tosa."), target, parents, attrs)
        else:
            raise ValueError(f"Unsupported operation: {name}")
        if node.shape != target:
            raise ValueError(f"Shape mismatch lowering {name}: {node.shape} != {target}")
        values[op.results[0]] = node
    if root is None:
        raise ValueError("Missing function return")
    return inputs, root


def fuse_relu(root):
    users = Counter(x for n in ordered(root) for x in n.inputs)
    replacements = {}
    fused = 0
    for node in ordered(root):
        node.inputs = tuple(replacements.get(x, x) for x in node.inputs)
        if node.kind == "relu":
            source = node.inputs[0]
            if source.kind not in ("conv2d", "add") or users[source] != 1:
                raise ValueError("ReLU requires a single-use convolution/add producer")
            source.attrs["relu"] = 1
            replacements[node] = source
            fused += 1
    return replacements.get(root, root), fused


def storage(node):
    while node.kind == "reshape":
        node = node.inputs[0]
    return node


def plan_memory(input_node, root, directory):
    nodes = ordered(root)
    counts = Counter()
    for node in nodes:
        node.name = f"{node.kind}_{counts[node.kind]}"
        counts[node.kind] += 1
    constants = [n for n in nodes if n.kind == "const"]
    cursor = 0
    with (directory / "parameters.bin").open("wb") as file:
        for node in constants:
            if node.data.dtype != np.float32:
                raise ValueError("Unexpected non-float parameter in executable graph")
            node.offset = cursor
            np.ascontiguousarray(node.data, dtype="<f4").tofile(file)
            cursor += node.elements * 8
    parameter_elements = cursor // 8
    input_node.offset = cursor
    cursor += input_node.elements * 8
    calls = [n for n in nodes if n.kind not in ("input", "const", "reshape")]
    last = {}
    for i, node in enumerate(calls):
        for parent in node.inputs:
            last[storage(parent)] = i
    last[storage(root)] = len(calls)
    free = []
    def release(offset, size):
        free.append((offset, size))
        free.sort()
        merged = []
        for offset, size in free:
            if merged and merged[-1][0] + merged[-1][1] == offset:
                merged[-1] = (merged[-1][0], merged[-1][1] + size)
            else:
                merged.append((offset, size))
        free[:] = merged
    for i, node in enumerate(calls):
        size = node.elements * 8
        fit = min(((s, j, o) for j, (o, s) in enumerate(free) if s >= size), default=None)
        if fit:
            capacity, index, node.offset = fit
            free.pop(index)
            if capacity > size:
                release(node.offset + size, capacity - size)
        else:
            node.offset = cursor
            cursor += size
        for parent in set(map(storage, node.inputs)):
            if parent.kind != "const" and last[parent] == i:
                release(parent.offset, parent.elements * 8)
    if cursor > (1 << 30) - 256:
        raise ValueError("Network arena exceeds the runtime interface limit")
    for node in nodes:
        if node.kind == "reshape":
            node.offset = storage(node).offset
    return nodes, calls, parameter_elements, cursor


def invocation(node):
    x = node.inputs[0]
    if node.kind == "conv2d":
        w, bias = node.inputs[1:]
        n, ih, iw, ic = x.shape
        oc, kh, kw, wic = w.shape
        on, oh, ow, ooc = node.shape
        pt, pb, pl, pr = node.attrs["pad"]
        sh, sw = node.attrs["stride"]
        dh, dw = node.attrs["dilation"]
        if min(sh, sw, dh, dw) <= 0 or min(pt, pb, pl, pr) < 0:
            raise ValueError("Invalid convolution padding/stride/dilation")
        expected = ((ih + pt + pb - (kh - 1) * dh - 1) // sh + 1,
                    (iw + pl + pr - (kw - 1) * dw - 1) // sw + 1)
        if (n, ic, oc, (oh, ow), bias.shape) != (on, wic, ooc, expected, (oc,)):
            raise ValueError("Incompatible convolution shapes")
        return "conv", [x.offset, w.offset, bias.offset, node.offset, n, ih, iw, ic,
                        oc, kh, kw, oh, ow, sh, sw, pt, pl, dh, dw, node.attrs.get("relu", 0)]
    if node.kind == "add":
        return "add", [x.offset, node.inputs[1].offset, node.offset, node.elements,
                       node.attrs.get("relu", 0)]
    if node.kind == "matmul":
        rhs = node.inputs[1]
        if len(x.shape) != 3 or x.shape[0] != 1 or rhs.shape[0] != 1:
            raise ValueError("Initial GEMM lowering requires batch one")
        _, m, k = x.shape
        _, bk, n = rhs.shape
        if bk != k or node.shape != (1, m, n):
            raise ValueError("Incompatible GEMM dimensions")
        return "gemm", [x.offset, rhs.offset, node.offset, m, n, k]
    if node.kind == "transpose":
        rank = len(x.shape)
        return "transpose", [x.offset, node.offset, rank, *x.shape, *([1] * (4-rank)),
                             *node.attrs["perm"], *range(rank, 4)]
    if node.kind in ("max_pool2d", "avg_pool2d"):
        n, ih, iw, ch = x.shape
        on, oh, ow, och = node.shape
        kh, kw = node.attrs["kernel"]
        sh, sw = node.attrs["stride"]
        pt, pb, pl, pr = node.attrs["pad"]
        if min(kh, kw, sh, sw) <= 0 or min(pt, pb, pl, pr) < 0:
            raise ValueError("Invalid pooling attributes")
        if (on, och, oh, ow) != (n, ch, (ih+pt+pb-kh)//sh+1, (iw+pl+pr-kw)//sw+1):
            raise ValueError("Incompatible pooling shape")
        if node.kind == "avg_pool2d" and any((pt, pb, pl, pr)):
            raise ValueError("Padded average pooling requires an explicit divisor policy")
        return "pool", [x.offset, node.offset, n, ih, iw, ch, oh, ow, kh, kw,
                        sh, sw, pt, pl, int(node.kind == "avg_pool2d")]
    raise ValueError(f"No invocation contract for {node.kind}")


def emit_host(input_node, root, calls, parameter_elements, arena_bytes, directory):
    constants, body, declarations, records = {}, [], {}, []
    def constant(n):
        if n not in constants:
            constants[n] = f"%c{len(constants)}"
            body.append(f"    {constants[n]} = arith.constant {n} : i64")
        return constants[n]
    def call(name, operands, attrs=""):
        declarations[name] = len(operands)
        body.append(f"    call @{name}({', '.join(operands)}){attrs} : "
                    f"({', '.join(['i64'] * len(operands))}) -> ()")
    body.append(f"    %ctx = call @esp_op_open({constant(arena_bytes)}, {constant(0)}) : (i64, i64) -> i64")
    for name, elements, offset in (("parameters", parameter_elements, 0),
                                    ("input", input_node.elements, input_node.offset)):
        body.append(f"    %{name}_view = memref.cast %{name} : memref<{elements}xf32> to memref<*xf32>")
        body.append(f"    call @esp_op_copy_in(%ctx, {constant(offset)}, %{name}_view) : (i64, i64, memref<*xf32>) -> ()")
    for index, node in enumerate(calls):
        kind, arguments = invocation(node)
        target = "accelerator" if kind in ("conv", "add", "gemm") else "cpu"
        attrs = (f' {{sodap.operator = "{kind}_f32_v1", sodap.instance = 0 : i64, '
                 f'sodap.invocation = {index} : i64, sodap.execution = "{target}"}}')
        call(f"esp_op_{kind}", ["%ctx", *[constant(x) for x in arguments]], attrs)
        records.append({"index": index, "operator": kind, "instance": 0,
                        "execution": target, "shape": node.shape, "arguments": arguments,
                        "inputs": [storage(x).name for x in node.inputs], "output": node.name,
                        "relu": bool(node.attrs.get("relu", 0))})
    body.append(f"    %output_view = memref.cast %output : memref<{root.elements}xf32> to memref<*xf32>")
    body.append(f"    call @esp_op_copy_out(%ctx, {constant(root.offset)}, %output_view) : (i64, i64, memref<*xf32>) -> ()")
    call("esp_op_close", ["%ctx"])
    body.append("    return")
    declarations = [f"  func.func private @{name}({', '.join(['i64'] * count)})"
                    for name, count in declarations.items()]
    declarations += ["  func.func private @esp_op_open(i64, i64) -> i64",
                     "  func.func private @esp_op_copy_in(i64, i64, memref<*xf32>)",
                     "  func.func private @esp_op_copy_out(i64, i64, memref<*xf32>)"]
    host = ("module {\n" + "\n".join(declarations) + "\n"
            f"  func.func @forward_kernel(%input: memref<{input_node.elements}xf32>, "
            f"%parameters: memref<{parameter_elements}xf32>, %output: memref<{root.elements}xf32>) {{\n"
            + "\n".join(body) + "\n  }\n}\n")
    checked = ir.Module.parse(host)
    checked.operation.verify()
    (directory / "host.mlir").write_text(str(checked))
    return records


def compile_network(source, directory):
    directory.mkdir(parents=True, exist_ok=True)
    with ir.Context():
        inputs, root = parse_graph(source.read_text())
        root, relus = fuse_relu(root)
        nodes, calls, parameter_elements, arena_bytes = plan_memory(inputs, root, directory)
        records = emit_host(inputs, root, calls, parameter_elements, arena_bytes, directory)
    plan = {"abi": 1, "input_shape": inputs.shape, "output_shape": root.shape,
            "arena_data_bytes": arena_bytes, "parameter_elements": parameter_elements,
            "fused_relus": relus, "operator_counts": dict(Counter(x["operator"] for x in records)),
            "bindings": {kind: {"instance": 0, "socket": 0, "module": f"{kind}_kernel"}
                         for kind in ("conv", "add", "gemm")},
            "parameters_sha256": hashlib.sha256((directory / "parameters.bin").read_bytes()).hexdigest(),
            "tensors": [{"name": n.name, "shape": n.shape, "offset_bytes": n.offset}
                        for n in nodes if n.kind != "reshape"], "invocations": records}
    (directory / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps({k: plan[k] for k in ("arena_data_bytes", "parameter_elements", "fused_relus", "operator_counts")}, indent=2))
    return plan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    options = parser.parse_args()
    compile_network(options.source, options.output)
