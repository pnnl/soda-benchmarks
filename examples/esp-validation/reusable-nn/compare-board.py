#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Compare successful CPU-only and accelerated inferences on the same ESP image."""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("cpu_result", type=Path)
parser.add_argument("fpga_result", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--cpu-hz", type=int, required=True)
args = parser.parse_args()
assert args.cpu_hz > 0

def load(path, execution):
    result = json.loads(path.read_text())
    assert result["status"] == "PASS" and result["execution"] == execution
    assert result["errors"] == 0 and result["cpu_cycles"] > 0
    for filename, expected in result["artifacts"].items():
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == expected, filename
    return result

cpu = load(args.cpu_result, "CPU_ONLY")
fpga = load(args.fpga_result, "FPGA")
assert cpu["soc"] == fpga["soc"] and cpu["jtag"] == fpga["jtag"]
assert cpu["outputs"] == fpga["outputs"] and cpu["operators"] == fpga["operators"]
assert cpu["hardware_invocations"] == 0
assert cpu["software_operator_invocations"] == fpga["hardware_invocations"]

def artifact(result, suffix):
    matches = [(Path(p), h) for p, h in result["artifacts"].items() if p.endswith(suffix)]
    assert len(matches) == 1, suffix
    return matches[0]

assert artifact(cpu, ".bit") == artifact(fpga, ".bit")
assert artifact(cpu, "/plan.json")[1] == artifact(fpga, "/plan.json")[1]
cpu_app = artifact(cpu, "/nn_app.exe")[0].parent
fpga_app = artifact(fpga, "/nn_app.exe")[0].parent
shared = {}
for filename in ("host.o", "input.bin", "parameters.bin", "reference.bin"):
    hashes = [hashlib.sha256((app / filename).read_bytes()).hexdigest()
              for app in (cpu_app, fpga_app)]
    assert hashes[0] == hashes[1], filename
    shared[filename] = hashes[0]

result = {
    "status": "PASS", "cpu_hz": args.cpu_hz,
    "cpu_only_cycles": cpu["cpu_cycles"], "accelerated_cycles": fpga["cpu_cycles"],
    "cpu_only_seconds": cpu["cpu_cycles"] / args.cpu_hz,
    "accelerated_seconds": fpga["cpu_cycles"] / args.cpu_hz,
    "accelerator_speedup": cpu["cpu_cycles"] / fpga["cpu_cycles"],
    "outputs": cpu["outputs"], "errors_cpu": 0, "errors_accelerated": 0,
    "cpu_max_error_e9": cpu["max_error_e9"],
    "accelerated_max_error_e9": fpga["max_error_e9"],
    "shared_inputs_sha256": shared,
    "cpu_result": str(args.cpu_result), "accelerated_result": str(args.fpga_result),
    "timing_scope": "forward_kernel including runtime setup, tensor copies and all graph operations; excludes boot, program loading and output validation",
    "cpu_baseline": "scalar reusable-operator C++ runtime; cacheable eight-byte tensor slots; no tuned BLAS",
    "samples_per_mode": 1,
}
args.output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
