#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Reuse a compiled ESP test SoC, rebuild its accelerator library and CPU binary.
set -euo pipefail
example=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$example/../../.." && pwd)
generated=$(realpath "${1:?Usage: run-system.sh extracted-gemm ESP-root compiled-SoC [run-root]}")
esp=$(realpath "${2:?ESP checkout required}")
soc=$(realpath "${3:?Compiled ESP SoC directory required}")
run=${4:-$(mktemp -d /tmp/esp-generated-gemm.XXXXXX)}
mkdir -p "$run/full-system" "$run/soft-build/ariane"
run=$(realpath "$run")
sim=$run/full-system
python3 "$example/make-vectors.py" "$sim"
grep -Eq 'CFG_L2_ENABLE[[:space:]]*: integer := 0;' "$soc/socgen/esp/esp_global.vhd"
grep -Eq 'CFG_LLC_ENABLE[[:space:]]*: integer := 0;' "$soc/socgen/esp/esp_global.vhd"
test -d "$soc/modelsim/work"
test ! -e "$sim/work" # Use a fresh run directory; never nest/overwrite a library.
cp -a "${ESP_MODELSIM_WORK:-$soc/modelsim/work}" "$sim/work"
cp "$soc/soft-build/ariane/prom.srec" "$sim/"
cp "$generated/bambu/array_429488.mem" "$sim/"
cat > "$sim/modelsim.ini" <<EOF
[Library]
others = ${ESP_MODELSIM_INI:-$soc/modelsim/modelsim.ini}
work = work
sqr_bambu = gemm_lib
EOF

cd "$sim"
vlib gemm_lib
vlog -sv -work sqr_bambu "$generated/bambu/forward_kernel.v" \
  "$generated/bambu/panda_libtech.v" "$example/sqr_bambu_basic_dma64.sv"
clang=${CLANG:-clang-16}
prefix=${RISCV_PREFIX:-riscv64-unknown-elf-}
"$clang" --target=riscv64-unknown-elf -march=rv64imafdc -mabi=lp64d \
  -mcmodel=medany -O2 -c "$generated/host.ll" -o host.o
drivers=$esp/soft/common/drivers
cpu=$esp/soft/ariane
libs=$soc/soft-build/ariane/drivers
flags=(-O2 -march=rv64imafdc -mabi=lp64d -mcmodel=medany -fno-common
  -fno-builtin-printf -ffp-contract=off
  -I"$sim"
  -I"$drivers/baremetal/include" -I"$drivers/common/include"
  -I"$cpu/riscv-tests/env" -I"$cpu/riscv-tests/benchmarks/common"
  -I"$cpu/bootrom" -I"$cpu/riscv-pk/machine" -I"$soc/socgen/esp")
"${prefix}gcc" "${flags[@]}" -std=gnu99 -c "$example/gemm_app.c" -o app.o
"${prefix}gcc" "${flags[@]}" -std=gnu99 -c "$cpu/common/syscalls.c" -o syscalls.o
"${prefix}gcc" "${flags[@]}" -c "$cpu/riscv-tests/benchmarks/common/crt.S" -o crt.o
"${prefix}g++" "${flags[@]}" -std=c++17 -fno-exceptions -fno-rtti \
  -DSODAP_ESP_DEVICE_ID=0x07a '-DSODAP_ESP_DEVICE_NAME="sld,sqr_bambu"' \
  -I"$repo/examples/soda-plugins/include" \
  -c "$repo/examples/soda-plugins/lib/sodap/ExecutionEngine/EspRuntime.cpp" -o runtime.o
"${prefix}gcc" "${flags[@]}" -static -nostdlib -nostartfiles \
  app.o runtime.o host.o syscalls.o crt.o \
  -T "$cpu/riscv-tests/benchmarks/common/test.ld" \
  "$libs/probe/libprobe.a" "$libs/monitors/libmonitors.a" \
  "$libs/utils/baremetal/libutils.a" -lm -lgcc -o gemm_app.exe
"${prefix}nm" --undefined-only gemm_app.exe > undefined-symbols.txt
test ! -s undefined-symbols.txt
"${prefix}objcopy" -O srec --gap-fill 0 gemm_app.exe ram.srec
"${prefix}objcopy" -O binary gemm_app.exe "$run/soft-build/ariane/systest.bin"
(cd "$run" && python3 "$esp/utils/scripts/file_handling/bin2txt_vhx.py" 64 ariane)
sha256sum "$generated/host.ll" "$generated/bambu/forward_kernel.v" \
  "$generated/bambu/panda_libtech.v" "$generated/bambu/array_429488.mem" \
  "$repo/examples/soda-plugins/lib/sodap/ExecutionEngine/EspRuntime.cpp" \
  "$repo/examples/soda-plugins/include/sodap/ExecutionEngine/ESPRuntime.h" \
  "$example/sqr_bambu_basic_dma64.sv" "$example/gemm_app.c" \
  vectors.h gemm_app.exe > inputs-and-software.sha256
cat > run.tcl <<'EOF'
onerror {quit -code 2 -f}
onbreak {quit -code 0 -f}
run 100 ms
puts "GEMM_SYSTEM_SIM_TIME_LIMIT"
quit -code 3 -f
EOF
vsim -c -suppress 3812,2697,8617,151,143,8386,3009,2685,2718 \
  -L secureip_ver -L unisims_ver -L work -L sqr_bambu \
  -uvmcontrol=disable -t fs +notimingchecks work.testbench work.glbl \
  -G/testbench/soc_inst/gen_mig_model/mig_axiram/kbytes="${ESP_SIM_RAM_KBYTES:-16384}" \
  -do 'do run.tcl' -l simulation.log +gemm_stall="${GEMM_STALL:-50000}" \
  > console.log 2>&1
python3 - <<'PY'
from pathlib import Path
import re
log = Path('simulation.log').read_text()
assert 'GEMM_SYSTEM PASS total_errors=0' in log
assert len(re.findall(r'GEMM_SYSTEM RESULT case=\d+ output_errors=0 input_errors=0', log)) == 3
assert '** Fatal' not in log and '** Error' not in log
assert 'GEMM_SYSTEM_SIM_TIME_LIMIT' not in log
print('PASS: three full-system GEMM invocations, 288 bit-exact outputs')
PY
printf 'Results: %s\n' "$sim/simulation.log"
