#!/usr/bin/env bash
#===----------------------------------------------------------------------===#
#
# Part of the SODA Benchmarks Project
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
#
#===----------------------------------------------------------------------===#
#
# Drives the `target_location` port that this recipe exposes on the top level.
#
# `target_location` is an is_global/is_extern input, so Bambu hoists it onto the
# generated top module and onto TestbenchDUT, but `bambu_testbench_impl`
# instantiates TestbenchDUT without connecting it -- the generated testbench only
# drives the kernel's own arguments. Verilator then reads it as 0, which is why
# an unpatched run always reports location 0.
#
# This hook adds the missing connection with a constant, so the simulation reads
# back the counter for the location of interest. That is the simulation stand-in
# for a real chip, where the port is driven externally and the core is re-run
# once per location.
#
# Override the selected location with SODA_TARGET_LOCATION, e.g.
#   make SODA_TARGET_LOCATION=1
# (exported through the environment; defaults to 0).

set -euo pipefail

BUILD_DIR="${1:-$PWD}"
TB="$BUILD_DIR/HLS_output/simulation/bambu_testbench.v"
TARGET_LOCATION="${SODA_TARGET_LOCATION:-0}"

if [ ! -f "$TB" ]; then
  echo "patch_tb.sh: testbench not found: $TB" >&2
  exit 1
fi

# Idempotency guard. Match only the constant this hook injects -- the file also
# contains TestbenchDUT's own `.target_location(target_location)` binding, which
# is the port being forwarded, not a driver.
if grep -q "\.target_location(32'd" "$TB"; then
  echo "patch_tb.sh: target_location already driven in $TB; nothing to do."
  exit 0
fi

# The DUT instantiation ends with the back-pressure port; append the connection
# just before its closing paren.
ANCHOR='    .Mout_back_pressure(sig_Mout_back_pressure));'
if ! grep -qF "$ANCHOR" "$TB"; then
  echo "patch_tb.sh: could not find the TestbenchDUT instantiation anchor in $TB." >&2
  echo "patch_tb.sh: Bambu's testbench layout may have changed; update this hook." >&2
  exit 1
fi

python3 - "$TB" "$TARGET_LOCATION" <<'PY'
import sys

tb_path, target_location = sys.argv[1], int(sys.argv[2])
anchor = "    .Mout_back_pressure(sig_Mout_back_pressure));"
replacement = (
    "    .Mout_back_pressure(sig_Mout_back_pressure),\n"
    f"    .target_location(32'd{target_location}));"
)

text = open(tb_path).read()
if text.count(anchor) != 1:
    sys.exit(
        f"patch_tb.sh: expected exactly one DUT instantiation, found {text.count(anchor)}"
    )
open(tb_path, "w").write(text.replace(anchor, replacement))
print(f"patch_tb.sh: driving target_location = {target_location}")
PY
