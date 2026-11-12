#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Prepend bad magic/version launches to captured GEMM traces, without reset.

Both invalid launches must leave the arena untouched; subsequent valid launches
must consume fresh descriptors and data, proving that all stage FIFOs drain.
"""
import argparse
from pathlib import Path
import re
import shutil

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("source", type=Path)
parser.add_argument("destination", type=Path)
args = parser.parse_args()
args.destination.mkdir(parents=True, exist_ok=False)
parameters = (args.source / "parameters.svh").read_text()
match = re.search(r"INVOCATIONS\s*=\s*(\d+)", parameters)
assert match, "Missing invocation count"
count = int(match[1])
parameters = parameters[:match.start(1)] + str(count + 2) + parameters[match.end(1):]
(args.destination / "parameters.svh").write_text(parameters)
initial = (args.source / "initial-0.hex").read_text().splitlines()
# ABI v1: magic and version are the low words of the first two DMA64 slots.
for invocation, word in enumerate((0, 2)):
    bad = initial.copy()
    bad[word] = f"{int(bad[word], 16) ^ 1:08x}"
    for name in ("initial", "expected"):
        (args.destination / f"{name}-{invocation}.hex").write_text("\n".join(bad) + "\n")
    (args.destination / f"mask-{invocation}.hex").write_text("0\n" * len(initial))
for invocation in range(count):
    for name in ("initial", "expected", "mask"):
        shutil.copyfile(args.source / f"{name}-{invocation}.hex",
                        args.destination / f"{name}-{invocation+2}.hex")
