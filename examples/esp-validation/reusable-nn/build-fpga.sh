#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Validate all three operators before building the complete network bitstream.
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
operators=$(realpath "${1:?Usage: build-fpga.sh operators-dir system-run-name isolated-SoC fresh-output}")
run=${2:?System run directory name required}
soc=$(realpath "${3:?Isolated SoC required}")
out=$(realpath -m "${4:?Fresh output required}")
for kind in conv add gemm; do
  sim="$operators/$kind/$run/full-system"
  python3 "$here/check-system.py" "$operators/$kind/esp-app/plan.json" "$sim/simulation.log"
  (cd "$sim" && sha256sum -c validation-inputs.sha256)
done
bash "$here/../generated-lcs/build-fpga.sh" "$soc" \
  "$operators/conv/$run/full-system/simulation.log" "$out"
