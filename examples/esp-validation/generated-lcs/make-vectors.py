#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Build DMA fixtures from the generated application ABI and PyTorch golden."""
import math
import re
import struct
import sys
from pathlib import Path

lcs, output = map(Path, sys.argv[1:])
output.mkdir(parents=True, exist_ok=True)
layout = dict(line.split("=", 1) for line in (lcs / "layout.txt").read_text().splitlines())
stride = int(layout["dma_word_stride"])
text = (lcs / "testdata.h").read_text()
if "typedef float td_elem_t;" not in text:
    sys.exit("the LCS fixture requires float32")
arrays = {}
for name, count, body in re.findall(
    r"static (?:const )?td_elem_t (\w+)\[(\d+)\](?:\s*=\s*\{(.*?)\})?;", text, re.S
):
    values = [float(v.strip()) for v in body.split(",") if v.strip()] if body else [0.0] * int(count)
    assert len(values) == int(count)
    arrays[name] = values
call = re.search(r"#define TD_CALL\(F\) F\((.*)\)", text)[1]
args = call.split(",")
signature = next(line for line in (lcs / "kernel.mlir").read_text().splitlines()
                 if "func.func @forward_kernel(" in line)
sizes = [math.prod(int(d) for d in dims.split("x") if d)
         for dims in re.findall(r"memref<((?:\d+x)*)f32>", signature)]
assert len(args) == len(sizes)
initial, expected, masks = [], [], []
for arg, count in zip(args, sizes):
    match = re.fullmatch(r"\s*\(?\s*(td_\w+)(?:\s*\+\s*(\d+))?\s*\)?\s*", arg)
    assert match, arg
    name, offset = match[1], int(match[2] or 0)
    values = arrays[name][offset:offset + count]
    gold = arrays["td_golden"][offset:offset + count] if name == "td_out" else values
    assert len(values) == count and len(gold) == count
    for value, target in zip(values, gold):
        initial.extend([value] + [0.0] * (stride - 1))
        expected.extend([target] + [0.0] * (stride - 1))
        masks.extend([2 if name == "td_out" else 1] + [1] * (stride - 1))
def bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]
for name, values in [("initial", initial), ("expected", expected)]:
    (output / f"{name}.hex").write_text("".join(f"{bits(v):08x}\n" for v in values))
(output / "mask.hex").write_text("".join(f"{v:x}\n" for v in masks))
tol = re.search(r"#define TD_TOL\s+(\S+)", text)[1]
(output / "parameters.svh").write_text(
    f"localparam WORDS = {len(initial)};\nlocalparam real TOL = {tol};\n")
