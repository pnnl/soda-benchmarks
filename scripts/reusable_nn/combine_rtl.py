#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Namespace complete Bambu engines, retaining all technology RTL and memories."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("hardware", type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
out = args.hardware / "combined"
out.mkdir(parents=True, exist_ok=True)
pieces, manifest = [], {}
token = re.compile(r'"(?:\\.|[^"\\])*"|//[^\n]*|/\*.*?\*/|\b[A-Za-z_][\w$]*\b', re.S)
for kind in ("conv", "add", "gemm"):
    directory = args.hardware / kind / "bambu_syn"
    files = [directory / f"{kind}_kernel.v", directory / "panda_libtech.v"]
    text = "\n".join(p.read_text() for p in files)
    names = re.findall(r"^module\s+([\w$]+)\b", text, re.M)
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate module inside {kind} engine")
    mapping = {name: f"{kind}__{name}" for name in names if name != f"{kind}_kernel"}
    # Bambu guards individual technology modules with preprocessor macros.
    # Namespace these too, or the first engine suppresses later definitions.
    mapping.update({name: f"{kind}__{name}" for name in re.findall(r"`define\s+(\w+)", text)})
    # Only rewrite identifier tokens, preserving strings and comments.
    text = token.sub(lambda m: mapping.get(m[0], m[0]), text)
    for memory in directory.glob("*.mem"):
        destination = f"{kind}__{memory.name}"
        text = text.replace(f'"{memory.name}"', f'"{destination}"')
        shutil.copyfile(memory, out / destination)
    pieces.append(text)
    manifest[kind] = {str(p.name): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(out / "operator_engines.v").write_text("\n".join(pieces))
shutil.copyfile(root / "examples/esp-validation/reusable-nn/operator_socket.sv", out / "forward_kernel.sv")
subprocess.run(["c++", "-std=c++17", "-I", str(root / "examples/soda-plugins/include"),
                str(root / "scripts/reusable_nn/print_abi.cpp"), "-o", str(out / "print-abi")], check=True)
with (out / "operator_abi.svh").open("w") as file:
    subprocess.run([str((out / "print-abi").resolve())], stdout=file, check=True)
(out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Combined one conv, one add and one GEMM engine in {out}")
