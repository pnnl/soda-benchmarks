#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
soc=$(realpath "${1:?Usage: build-fpga.sh isolated-SoC simulation-log output-directory}")
simulation=$(realpath "${2:?Completed full-system simulation log required}")
output=$(realpath -m "${3:?Output directory required}")
grep -q 'TEST PASSED' "$simulation"
grep -q 'Failure: Program Completed!' "$simulation"
! grep -Eq 'TEST FAILED|LCS_SYSTEM_SIM_TIME_LIMIT|\*\* Fatal|\*\* Error' "$simulation"
test -f "$soc/../../integration.json"
test ! -f "$soc/vivado/esp-xilinx-vc707-xc7vx485t.xpr"
mkdir -p "$output"
if [[ -z ${XILINXD_LICENSE_FILE:-} ]]; then
  setup=${XILINX_ENV_SCRIPT:-/opt/cad/scripts/xilinx_env.sh}
  if [[ -r $setup ]]; then
    set +u
    CAD_PATH=${CAD_PATH:-/opt/cad} source "$setup"
    set -u
  fi
fi
export XILINX_VIVADO=${XILINX_VIVADO:-/opt/xilinx/Vivado/2023.2}
export PATH="$XILINX_VIVADO/bin:/opt/riscv/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1
cd "$soc"
make vivado-setup > "$output/project-setup.log" 2>&1
project=$soc/vivado/esp-xilinx-vc707-xc7vx485t.xpr
test -s "$project"
cd "$soc/vivado"
vivado -mode batch -nojournal -log "$output/vivado.log" \
  -source "$here/implement.tcl" -tclargs "$project" "$output/reports" "${VIVADO_JOBS:-4}" \
  > "$output/implementation.log" 2>&1
test -s "$soc/vivado/esp-xilinx-vc707-xc7vx485t.runs/impl_1/top.bit"
cp "$soc/vivado/esp-xilinx-vc707-xc7vx485t.runs/impl_1/top.bit" "$output/top.bit"
ln -s vivado/esp-xilinx-vc707-xc7vx485t.runs/impl_1/top.bit "$soc/top.bit"
sha256sum "$output/top.bit" > "$output/bitstream.sha256"
python3 - "$output/reports/timing.tsv" <<'PY'
import sys
from pathlib import Path
rows = dict(line.split('\t') for line in Path(sys.argv[1]).read_text().splitlines()[1:])
for name in ('STATS.WNS', 'STATS.WHS'):
    assert float(rows[name]) >= 0, f'Timing failure: {name}={rows[name]}'
print('PASS: bitstream generated and setup/hold slack nonnegative')
PY
