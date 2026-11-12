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
import struct
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
    for output in roots(root):
        visit(output)
    return result


def roots(root):
    return root if isinstance(root, (list, tuple)) else [root]


def reshape(x, target):
    target = tuple(target)
    if x.elements != math.prod(target):
        raise ValueError("Reshape changes the number of elements")
    if x.shape == target:
        return x
    if x.kind == "const":
        return Node("const", target, data=x.data.reshape(target))
    if x.kind == "reshape":
        return reshape(x.inputs[0], target)
    if x.kind == "pad":
        old = [i for i,d in enumerate(x.shape) if d != 1]
        new = [i for i,d in enumerate(target) if d != 1]
        if [x.shape[i] for i in old] == [target[i] for i in new]:
            pads = [(0,0)] * len(target)
            for a,b in zip(old,new):
                pads[b] = x.attrs["pads"][a]
            unpadded = tuple(d-l-r for d,(l,r) in zip(target,pads))
            return Node("pad",target,(reshape(x.inputs[0],unpadded),),{"pads":tuple(pads)})
    return Node("reshape", target, (x,))


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
    if x.kind == "pad":
        return Node("pad", tuple(x.shape[i] for i in p), (transpose(x.inputs[0],p),),
                    {"pads":tuple(x.attrs["pads"][i] for i in p)})
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


def unary_activation(kind, x, attrs):
    if x.kind == "transpose":
        return transpose(unary_activation(kind,x.inputs[0],attrs), x.attrs["perm"])
    if x.kind == "reshape":
        return reshape(unary_activation(kind,x.inputs[0],attrs), x.shape)
    return Node(kind,x.shape,(x,),attrs)


def scalar(node):
    if node.kind == "const" and node.elements == 1:
        return float(node.data.reshape(-1)[0])
    return None


def leaky_pattern(a,b):
    # Torch-MLIR: max(x,0) + min(x,0)*alpha. Never treat arbitrary mul/add as an activation.
    for maximum, product in ((a,b),(b,a)):
        if maximum.kind != "maximum" or product.kind != "mul":
            continue
        for x,zero in (maximum.inputs,maximum.inputs[::-1]):
            if scalar(zero) != 0:
                continue
            for minimum,coefficient in (product.inputs,product.inputs[::-1]):
                alpha = scalar(coefficient)
                if minimum.kind != "minimum" or alpha is None or not math.isfinite(alpha) or not 0 <= alpha <= 1:
                    continue
                if any(y is x and scalar(z) == 0 for y,z in (minimum.inputs,minimum.inputs[::-1])):
                    return unary_activation("leaky_relu",x,{"alpha":alpha})
    return None


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
            if not op.operands:
                raise ValueError("Expected at least one output tensor")
            outputs = [values[v] for v in op.operands]
            root = outputs[0] if len(outputs) == 1 else outputs
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
            node = reshape(parents[0],target)
        elif name == "tosa.pad":
            if len(parents) != 3 or parents[1].kind != "const" or scalar(parents[2]) != 0:
                raise ValueError("Only constant zero padding can be folded into convolution")
            pads = tuple(tuple(map(int,row)) for row in parents[1].data)
            if len(pads) != len(target) or any(len(p)!=2 or min(p)<0 for p in pads):
                raise ValueError("Invalid zero padding")
            expected = tuple(d+l+r for d,(l,r) in zip(parents[0].shape,pads))
            node = Node("pad",expected,(parents[0],),{"pads":pads})
        elif name == "tosa.clamp":
            lo = ir.FloatAttr(op.attributes["min_fp"]).value
            hi = ir.FloatAttr(op.attributes["max_fp"]).value
            if lo != 0 or hi != float(np.finfo(np.float32).max):
                raise ValueError("Only the float32 ReLU clamp is supported")
            node = unary_activation("relu",parents[0],{})
        elif name in ("tosa.maximum", "tosa.minimum", "tosa.mul"):
            if name == "tosa.mul" and int(ir.IntegerAttr(op.attributes["shift"]).value) != 0:
                raise ValueError("LeakyReLU requires unshifted multiplication")
            node = Node(name.removeprefix("tosa."),target,parents)
        elif name == "tosa.add":
            node = leaky_pattern(*parents)
            if node is None:
                node = pointwise("add", parents, {})
        elif name in ("tosa.conv2d", "tosa.matmul", "tosa.max_pool2d", "tosa.avg_pool2d"):
            if name == "tosa.conv2d" and parents[0].kind == "pad":
                padded = parents[0]
                pads = padded.attrs["pads"]
                if len(pads) != 4 or pads[0] != (0,0) or pads[3] != (0,0):
                    raise ValueError("Convolution can only absorb spatial zero padding")
                attrs["pad"] = tuple(a+b for a,b in zip(attrs["pad"],(*pads[1],*pads[2])))
                parents = (padded.inputs[0],*parents[1:])
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
    users.update(roots(root))
    replacements = {}
    fused = 0
    for node in ordered(root):
        node.inputs = tuple(replacements.get(x, x) for x in node.inputs)
        if node.kind in ("relu", "leaky_relu"):
            source = node.inputs[0]
            supported = ("conv2d",) if node.kind == "leaky_relu" else ("conv2d", "add")
            if source.kind not in supported or users[source] != 1 or "activation" in source.attrs or "relu" in source.attrs:
                raise ValueError("Activation requires a single-use supported convolution/add producer")
            if node.kind == "leaky_relu":
                source.attrs.update(activation="leaky_relu",alpha=node.attrs["alpha"])
            else:
                source.attrs["relu"] = 1
            replacements[node] = source
            fused += int(node.kind == "relu")
    result = [replacements.get(x,x) for x in roots(root)]
    return (result if isinstance(root,(list,tuple)) else result[0]), fused


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
    for output in roots(root):
        last[storage(output)] = len(calls)
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
        arguments = [x.offset,w.offset,bias.offset,node.offset,n,ih,iw,ic,oc,kh,kw,oh,ow,sh,sw,pt,pl,dh,dw]
        if node.attrs.get("activation") == "leaky_relu":
            arguments += [2,struct.unpack("<I",struct.pack("<f",node.attrs["alpha"]))[0]]
        else:
            arguments += [node.attrs.get("relu",0)]
        return "conv", arguments
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
    output_elements = sum(x.elements for x in roots(root))
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
        version = 2 if kind == "conv" and node.attrs.get("activation") == "leaky_relu" else 1
        attrs = (f' {{sodap.operator = "{kind}_f32_v{version}", sodap.instance = 0 : i64, '
                 f'sodap.invocation = {index} : i64, sodap.execution = "{target}"}}')
        function = "esp_op_conv_activation" if kind == "conv" and node.attrs.get("activation") == "leaky_relu" else f"esp_op_{kind}"
        call(function, ["%ctx", *[constant(x) for x in arguments]], attrs)
        records.append({"index": index, "operator": kind, "instance": 0,
                        "execution": target, "shape": node.shape, "arguments": arguments,
                        "inputs": [storage(x).name for x in node.inputs], "output": node.name,
                        "relu": bool(node.attrs.get("relu", 0)),
                        "activation":node.attrs.get("activation","relu" if node.attrs.get("relu") else "none"),
                        "alpha":node.attrs.get("alpha")})
        if kind == "conv":
            names = ["a_bytes","w_bytes","bias_bytes","c_bytes","n","ih","iw","ic","oc","kh","kw",
                     "oh","ow","stride_h","stride_w","pad_top","pad_left","dilation_h","dilation_w","activation_mode","alpha_f32_bits"]
            records[-1]["configuration"] = dict(zip(names,arguments))
            records[-1]["descriptor_version"] = 2 if len(arguments)==21 else 1
    output_offset = 0
    for i,output in enumerate(roots(root)):
        view_type = f"memref<{output.elements}xf32, strided<[1], offset: {output_offset}>>"
        body.append(f"    %out{i} = memref.subview %output[{output_offset}] [{output.elements}] [1] : memref<{output_elements}xf32> to {view_type}")
        body.append(f"    %output_view{i} = memref.cast %out{i} : {view_type} to memref<*xf32>")
        body.append(f"    call @esp_op_copy_out(%ctx, {constant(output.offset)}, %output_view{i}) : (i64, i64, memref<*xf32>) -> ()")
        output_offset += output.elements
    call("esp_op_close", ["%ctx"])
    body.append("    return")
    declarations = [f"  func.func private @{name}({', '.join(['i64'] * count)})"
                    for name, count in declarations.items()]
    declarations += ["  func.func private @esp_op_open(i64, i64) -> i64",
                     "  func.func private @esp_op_copy_in(i64, i64, memref<*xf32>)",
                     "  func.func private @esp_op_copy_out(i64, i64, memref<*xf32>)"]
    host = ("module {\n" + "\n".join(declarations) + "\n"
            f"  func.func @forward_kernel(%input: memref<{input_node.elements}xf32>, "
            f"%parameters: memref<{parameter_elements}xf32>, %output: memref<{output_elements}xf32>) {{\n"
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
    outputs = roots(root)
    plan = {"abi": 1, "input_shape": inputs.shape,
            "output_shape":outputs[0].shape if len(outputs)==1 else (sum(x.elements for x in outputs),),
            "outputs":[{"shape":x.shape,"offset_elements":sum(y.elements for y in outputs[:i]),
                        "arena_offset_bytes":x.offset} for i,x in enumerate(outputs)],
            "arena_data_bytes": arena_bytes, "parameter_elements": parameter_elements,
            "fused_relus": relus, "operator_counts": dict(Counter(x["operator"] for x in records)),
            "fused_leaky_relus":sum(x.attrs.get("activation")=="leaky_relu" for x in nodes),
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
