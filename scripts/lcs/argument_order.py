#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Extract the call ABI from the exact module used by both LCS pipelines."""
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text()
launches = [
    line for line in text.splitlines()
    if re.search(
        r"soda\.launch_func\s+@forward_kernel(?:::?@forward_kernel)?\s+args\(", line
    )
]
if len(launches) != 1:
    sys.exit("expected exactly one soda.launch_func @forward_kernel")
arguments = launches[0].split("args(", 1)[1]
order = re.findall(r"%arg(\d+)\s*:", arguments)
if not order or len(order) != len(set(order)):
    sys.exit("unsupported outlined argument mapping; expected distinct forward arguments")
print(" ".join(order))
