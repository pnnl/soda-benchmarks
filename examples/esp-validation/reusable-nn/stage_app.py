#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Stage the same compiled network and binary tensors for ESP baremetal."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("model", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
plan = json.loads((args.model/"compiled/plan.json").read_text())
for name, shape in (("input", plan["input_shape"]), ("reference", plan["output_shape"])):
    data = np.load(args.model/f"{name}.npy")
    assert data.size == np.prod(shape)
    np.asarray(data, dtype="<f4").tofile(args.output/f"{name}.bin")
shutil.copyfile(args.model/"compiled/parameters.bin", args.output/"parameters.bin")
shutil.copyfile(args.model/"compiled/host-riscv.o", args.output/"host.o")
shutil.copyfile(args.model/"compiled/plan.json", args.output/"plan.json")
# Relative incbin paths are resolved by building in this directory. Avoid huge
# generated C initializers and keep the weight bytes identical to native runs.
assembly = ['.section .rodata.nn,"a",@progbits']
for name in ("input", "parameters", "reference"):
    assembly += ['.balign 64', f'.global nn_{name}', f'nn_{name}:', f'.incbin "{name}.bin"']
(args.output/"tensors.S").write_text("\n".join(assembly)+"\n")
(args.output/"model.h").write_text(
    f'#define NN_OUTPUT_ELEMENTS {np.prod(plan["output_shape"])}\n'
    f'#define NN_ARENA_BYTES {plan["arena_data_bytes"]+256}\n')
