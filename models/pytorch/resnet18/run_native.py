#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Execute generated host LLVM with the reusable operator runtime."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import time
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("directory", type=Path)
parser.add_argument("--library", default="native.so")
parser.add_argument("--backend", default="native-operator-oracle")
parser.add_argument("--runtime", type=Path, help="Load a separate RTL runtime before the host-only library")
parser.add_argument("--atol", type=float, default=1e-3)
parser.add_argument("--rtol", type=float, default=1e-4)
args = parser.parse_args()
args.directory = args.directory.resolve()
plan = json.loads((args.directory / "compiled/plan.json").read_text())
data = np.ascontiguousarray(np.load(args.directory / "input.npy"), dtype=np.float32)
parameters = np.fromfile(args.directory / "compiled/parameters.bin", dtype="<f4")
reference = np.load(args.directory / "reference.npy")
assert data.shape == tuple(plan["input_shape"])
assert parameters.size == plan["parameter_elements"]
output = np.empty(plan["output_shape"], dtype=np.float32)
if args.runtime:
    runtime_path = args.runtime.resolve()
    runtime = ctypes.CDLL(str(runtime_path), mode=ctypes.RTLD_GLOBAL)
    os.chdir(runtime_path.parent)  # Generated RTL memory initialization files.
library = ctypes.CDLL(str((args.directory / "compiled" / args.library).resolve()))
kernel = library.forward_kernel
pointer = ctypes.POINTER(ctypes.c_float)
kernel.argtypes = [pointer, pointer, pointer]
kernel.restype = None
start = time.monotonic()
kernel(*(x.ctypes.data_as(pointer) for x in (data, parameters, output)))
elapsed = time.monotonic() - start
np.save(args.directory / f"{args.backend}-output.npy", output)
reference = reference.reshape(output.shape)
assert np.isfinite(output).all() and np.isfinite(reference).all()
np.testing.assert_allclose(output, reference, atol=args.atol, rtol=args.rtol)
summary = {"status": "PASS", "backend": args.backend, "outputs": output.size,
           "max_abs_error": float(np.max(np.abs(output-reference))), "atol": args.atol, "rtol": args.rtol,
           "wall_seconds": elapsed, "top5": np.argsort(output.reshape(-1))[-5:][::-1].tolist(),
           "reference_top5": np.argsort(reference.reshape(-1))[-5:][::-1].tolist(),
           "operator_counts": plan["operator_counts"], "fused_relus": plan["fused_relus"],
           "fused_leaky_relus":plan.get("fused_leaky_relus",0), "output_slices":plan.get("outputs",[])}
(args.directory / f"{args.backend}-validation.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
