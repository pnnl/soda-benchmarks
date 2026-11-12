#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
hardware=$(realpath "${1:?Usage: run-system.sh hardware-dir staged-app compiled-SoC fresh-run-dir}")
app=$(realpath "${2:?Staged and linked application required}")
soc=$(realpath "${3:?Compiled SoC required}")
run=$(realpath -m "${4:?Fresh run directory required}")
test ! -e "$run"
test -s "$app/nn_app.exe"
grep -Eq 'CFG_L2_ENABLE[[:space:]]*: integer := 0;' "$soc/socgen/esp/esp_global.vhd"
grep -Eq 'CFG_LLC_ENABLE[[:space:]]*: integer := 0;' "$soc/socgen/esp/esp_global.vhd"
mkdir -p "$run/full-system" "$run/soft-build/ariane"
sim=$run/full-system
cp -a "${ESP_MODELSIM_WORK:-$soc/modelsim/work}" "$sim/work"
cp "$soc/soft-build/ariane/prom.srec" "$app/ram.srec" "$sim/"
cp "$app/nn_app.bin" "$run/soft-build/ariane/systest.bin"
python3 - "$app/nn_app.bin" "$run/soft-build/ariane/ram.vhx" <<'PY'
from pathlib import Path
import sys
data=Path(sys.argv[1]).read_bytes()
data=data.ljust(max(2*1024**2,(len(data)+7)//8*8),b'\0')
Path(sys.argv[2]).write_text(''.join(data[i:i+8].hex()+'\n' for i in range(0,len(data),8)))
PY
shopt -s nullglob
memories=("$hardware/combined/"*.mem)
if ((${#memories[@]})); then cp "${memories[@]}" "$sim/"; fi
cat > "$sim/modelsim.ini" <<EOF
[Library]
others = ${ESP_MODELSIM_INI:-$soc/modelsim/modelsim.ini}
work = work
sqr_bambu = operator_lib
EOF
cd "$sim"
if [[ ${ESP_RECOMPILE_SOC_TOP:-0} == 1 ]]; then
  vcom -93 -work work "$soc/../../rtl/techmap/virtex7/pads_unisim.vhd" \
    "$soc/../../rtl/techmap/virtex7/clkgen_unisim.vhd" \
    "$soc/../../rtl/techmap/virtex7/memory_unisim.vhd" \
    "$soc/../../rtl/sim/models/axi_ram_sim_wrapper.vhd" "$soc/top.vhd" "$soc/testbench.vhd"
fi
vlib operator_lib
vlog -sv +define+SODAP_LCS_GENERIC +incdir+"$hardware/combined" -work sqr_bambu \
  "$hardware/combined/operator_engines.v" "$hardware/combined/forward_kernel.sv" \
  "$here/../generated-gemm/sqr_bambu_basic_dma64.sv"
sim_time_ms=${ESP_SIM_TIME_MS:-100}
[[ "$sim_time_ms" =~ ^[1-9][0-9]*$ ]] || exit 1
timeout_cycles=${NN_TIMEOUT_CYCLES:-2000000}
[[ "$timeout_cycles" =~ ^[1-9][0-9]*$ ]] && ((timeout_cycles < 100000000)) || exit 1
ram_kbytes=${ESP_SIM_RAM_KBYTES:-16384}
python3 - "$app/memory.json" "$ram_kbytes" <<'PY'
import json,sys
assert json.load(open(sys.argv[1]))['dram_bytes']==int(sys.argv[2])*1024, 'Software/simulator DRAM mismatch'
PY
ln -s /dev/null trace_hart_0.log
ln -s /dev/null trace_hart_0_commit.log
cat > run.tcl <<EOF
onerror {quit -code 2 -f}
onbreak {quit -code 0 -f}
run $sim_time_ms ms
puts "NN_SYSTEM_SIM_TIME_LIMIT"
quit -code 3 -f
EOF
vsim -c -suppress 3812,2697,8617,151,143,8386,3009,2685,2718 \
  -L secureip_ver -L unisims_ver -L work -L sqr_bambu \
  -uvmcontrol=disable -t fs +notimingchecks work.testbench work.glbl \
  -G/testbench/soc_inst/gen_mig_model/mig_axiram/kbytes="$ram_kbytes" \
  -do 'do run.tcl' -l simulation.log +gemm_stall="${NN_STALL:-0}" \
  +gemm_timeout="$timeout_cycles" > console.log 2>&1
python3 "$here/check-system.py" "$app/plan.json" simulation.log
sha256sum "$app/nn_app.exe" "$app/plan.json" \
  "$hardware/combined/operator_engines.v" "$hardware/combined/forward_kernel.sv" \
  "$hardware/combined/operator_abi.svh" "$here/../generated-gemm/sqr_bambu_basic_dma64.sv" \
  > validation-inputs.sha256
printf 'Results: %s\n' "$sim/validation.json"
