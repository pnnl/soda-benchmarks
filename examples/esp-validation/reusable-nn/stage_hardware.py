#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Package the combined socket for the existing isolated ESP FPGA flow."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("hardware",type=Path)
parser.add_argument("output",type=Path)
args=parser.parse_args()
combined=args.hardware/"combined"
rtl=args.output/"bambu_syn"
rtl.mkdir(parents=True,exist_ok=True)
wrapper=(combined/"forward_kernel.sv").read_text()
assert wrapper.count('`include "operator_abi.svh"')==1
wrapper=wrapper.replace('`include "operator_abi.svh"',(combined/"operator_abi.svh").read_text())
(rtl/"forward_kernel.v").write_text((combined/"operator_engines.v").read_text()+"\n"+wrapper)
(rtl/"panda_libtech.v").write_text("// Technology modules are namespaced inside forward_kernel.v.\n")
for memory in combined.glob("*.mem"):
    shutil.copyfile(memory,rtl/memory.name)
(args.output/"manifest.json").write_text(json.dumps({
    "instances":{"conv":1,"add":1,"gemm":1},"esp_sockets":1,
    "source_sha256":{name:hashlib.sha256((combined/name).read_bytes()).hexdigest()
                     for name in ("operator_engines.v","forward_kernel.sv","operator_abi.svh")},
    "packaged_rtl_sha256":hashlib.sha256((rtl/"forward_kernel.v").read_bytes()).hexdigest()
},indent=2)+"\n")
