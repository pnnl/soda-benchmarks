#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Reuse a compiled ESP test SoC, rebuild its accelerator library and CPU binary.
set -euo pipefail
example=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$example/../../.." && pwd)
generated=$(realpath "${1:?Usage: run-system.sh output/esp/<flow> ESP-root compiled-SoC [run-root]}")
wrapper=$(python3 "$repo/scripts/lcs/native_dma_contract.py" select "$generated" --wrapper "${LCS_WRAPPER:-thin}")
esp=$(realpath "${2:?ESP checkout required}")
soc=$(realpath "${3:?Compiled ESP SoC directory required}")
run=${4:-$(mktemp -d /tmp/esp-generated-gemm.XXXXXX)}
mkdir -p "$run/full-system" "$run/soft-build/ariane"
run=$(realpath "$run")
sim=$run/full-system
mode=${LCS_SYSTEM_MODE:-esp}
case "$mode" in esp|cpu-baseline) ;; *) echo "LCS_SYSTEM_MODE must be esp or cpu-baseline" >&2; exit 1 ;; esac
artifact_dir="$generated"
if [[ "$mode" == cpu-baseline ]]; then artifact_dir="$generated/cpu-baseline"; fi
app="$artifact_dir/esp-app"
appname=$(sed -n 's/^APPNAME *:= *//p' "$app/Makefile")
cp "$app/testdata.h" "$sim/"
shopt -s nullglob
grep -Eq 'CFG_L2_ENABLE[[:space:]]*: integer := 0;' "$soc/socgen/esp/esp_global.vhd"
grep -Eq 'CFG_LLC_ENABLE[[:space:]]*: integer := 0;' "$soc/socgen/esp/esp_global.vhd"
test -d "$soc/modelsim/work"
test ! -e "$sim/work" # Use a fresh run directory; never nest/overwrite a library.
cp -a "${ESP_MODELSIM_WORK:-$soc/modelsim/work}" "$sim/work"
cp "$soc/soft-build/ariane/prom.srec" "$sim/"
memories=("$generated/bambu_syn/"*.mem)
if ((${#memories[@]})); then cp "${memories[@]}" "$sim/"; fi
cat > "$sim/modelsim.ini" <<EOF
[Library]
others = ${ESP_MODELSIM_INI:-$soc/modelsim/modelsim.ini}
work = work
sqr_bambu = gemm_lib
EOF

cd "$sim"
if [[ ${ESP_RECOMPILE_SOC_TOP:-0} == 1 ]]; then
  # Refresh dependencies when the vendor simulation libraries were rebuilt.
  vcom -93 -work work "$soc/../../rtl/techmap/virtex7/pads_unisim.vhd" \
    "$soc/../../rtl/techmap/virtex7/clkgen_unisim.vhd" \
    "$soc/../../rtl/techmap/virtex7/memory_unisim.vhd" \
    "$soc/../../rtl/sim/models/axi_ram_sim_wrapper.vhd" \
    "$soc/top.vhd" "$soc/testbench.vhd"
fi
vlib gemm_lib
vlog -sv +define+SODAP_LCS_GENERIC -work sqr_bambu "$generated/bambu_syn/forward_kernel.v" \
  "$generated/bambu_syn/panda_libtech.v" "$wrapper"
prefix=${RISCV_PREFIX:-riscv64-unknown-elf-}
cp "$artifact_dir/06_kernel_riscv.o" host.o
drivers=$esp/soft/common/drivers
cpu=$esp/soft/ariane
libs=$soc/soft-build/ariane/drivers
flags=(-O2 -march=rv64imafdc -mabi=lp64d -mcmodel=medany -fno-common
  -fno-builtin-printf -ffp-contract=off
  -I"$sim" -I"$repo/examples/soda-plugins/include/sodap/ExecutionEngine"
  -I"$drivers/baremetal/include" -I"$drivers/common/include"
  -I"$cpu/riscv-tests/env" -I"$cpu/riscv-tests/benchmarks/common"
  -I"$cpu/bootrom" -I"$cpu/riscv-pk/machine" -I"$soc/socgen/esp")
"${prefix}gcc" "${flags[@]}" -std=gnu99 -c "$app/$appname.c" -o app.o
"${prefix}gcc" "${flags[@]}" -std=gnu99 -c "$repo/examples/soda-plugins/lib/sodap/ExecutionEngine/esp_prof.c" -o profiler.o
"${prefix}gcc" "${flags[@]}" -std=gnu99 -c "$repo/scripts/lib/memref_copy.c" -o memref_copy.o
"${prefix}gcc" "${flags[@]}" -std=gnu99 -c "$cpu/common/syscalls.c" -o syscalls.o
"${prefix}gcc" "${flags[@]}" -D_init=esp_init_bss -c "$cpu/riscv-tests/benchmarks/common/crt.S" -o crt.o
"${prefix}gcc" "${flags[@]}" -c "$example/baremetal-startup.S" -o startup.o
"${prefix}g++" "${flags[@]}" -std=c++17 -fno-exceptions -fno-rtti \
  -DSODAP_ESP_DEVICE_ID=0x07a '-DSODAP_ESP_DEVICE_NAME="sld,sqr_bambu"' \
  -I"$repo/examples/soda-plugins/include" \
  -c "$repo/examples/soda-plugins/lib/sodap/ExecutionEngine/EspRuntime.cpp" -o runtime.o
extra_runtime=()
if grep -q 'ReusableGemmRuntime.o:' "$app/Makefile"; then
  "${prefix}g++" "${flags[@]}" -std=c++17 -fno-exceptions -fno-rtti \
    -DSODAP_ESP_DEVICE_ID=0x07a '-DSODAP_ESP_DEVICE_NAME="sld,sqr_bambu"' \
    -I"$repo/examples/soda-plugins/include" \
    -c "$repo/examples/soda-plugins/lib/sodap/ExecutionEngine/ReusableGemmRuntime.cpp" \
    -o reusable-runtime.o
  extra_runtime+=(reusable-runtime.o)
fi
# Explicitly place sections that GNU ld used to insert as orphans. ESP's
# boot ROM jumps to 0x80000000; LLD otherwise places .rodata ahead of _start.
python3 - "$cpu/riscv-tests/benchmarks/common/test.ld" <<'LINKER'
from pathlib import Path
import sys
s = Path(sys.argv[1]).read_text()
s = s.replace('.text : { *(.text) }', '.text : { *(.text .text.*) }\n  .rodata : { *(.rodata .rodata.*) }\n  /DISCARD/ : { *(.eh_frame .eh_frame.*) }')
s = s.replace('.data : { *(.data) }', '.data : { *(.data .data.*) }')
assert s.count('.sbss : {') == 1 and s.count('.bss : { *(.bss) }') == 1
s = s.replace('.sbss : {', '.sbss : {\n    . = ALIGN(8);\n    __esp_bss_start = .;')
s = s.replace('.bss : { *(.bss) }', '.bss : { *(.bss .bss.* COMMON) . = ALIGN(8); __esp_bss_end = .; }')
Path('test.ld').write_text(s)
LINKER
# Older ESP GNU linkers cannot parse Clang 19's Zicsr/Zmmul ISA attributes.
# LLD links the original object without stripping architecture information.
"${LD_LLD:-ld.lld}" -m elf64lriscv -static \
  app.o runtime.o "${extra_runtime[@]}" host.o profiler.o memref_copy.o syscalls.o crt.o startup.o \
  -T test.ld \
  "$libs/probe/libprobe.a" "$libs/monitors/libmonitors.a" \
  "$libs/utils/baremetal/libutils.a" \
  "$("${prefix}gcc" -march=rv64imafdc -mabi=lp64d -print-file-name=libm.a)" \
  "$("${prefix}gcc" -march=rv64imafdc -mabi=lp64d -print-file-name=libgcc.a)" \
  -o gemm_app.exe

"${prefix}readelf" -h gemm_app.exe > elf-header.txt
grep -Eq 'Entry point address: +0x80000000' elf-header.txt
"${prefix}nm" --undefined-only gemm_app.exe > undefined-symbols.txt
test ! -s undefined-symbols.txt
"${prefix}objcopy" -O srec --gap-fill 0 gemm_app.exe ram.srec
"${prefix}objcopy" -O binary gemm_app.exe "$run/soft-build/ariane/systest.bin"
(cd "$run" && python3 "$esp/utils/scripts/file_handling/bin2txt_vhx.py" 64 ariane)
sha256sum "$artifact_dir/06_kernel_riscv.o" "$generated/bambu_syn/forward_kernel.v" \
  "$generated/bambu_syn/panda_libtech.v" "$generated/layout.txt" \
  "$repo/examples/soda-plugins/lib/sodap/ExecutionEngine/EspRuntime.cpp" \
  "$repo/examples/soda-plugins/include/sodap/ExecutionEngine/ESPRuntime.h" \
  "$wrapper" "$app/$appname.c" \
  "$example/baremetal-startup.S" test.ld testdata.h gemm_app.exe > inputs-and-software.sha256
if ((${#extra_runtime[@]})); then
  sha256sum "$repo/examples/soda-plugins/lib/sodap/ExecutionEngine/ReusableGemmRuntime.cpp" \
    "$repo/examples/soda-plugins/include/sodap/ExecutionEngine/ReusableGemmABI.h" \
    >> inputs-and-software.sha256
fi
if [[ ${ESP_BUILD_ONLY:-0} == 1 ]]; then
  echo 'BUILT: baremetal application linked; simulation not run'
  exit 0
fi
sim_time_ms=${ESP_SIM_TIME_MS:-100}
[[ "$sim_time_ms" =~ ^[1-9][0-9]*$ ]] || { echo "ESP_SIM_TIME_MS must be a positive integer" >&2; exit 1; }
timeout_cycles=${LCS_TIMEOUT_CYCLES:-1000000}
[[ "$timeout_cycles" =~ ^[1-9][0-9]*$ ]] && ((timeout_cycles < 100000000)) || {
  echo 'LCS_TIMEOUT_CYCLES must be between 1 and 99999999' >&2; exit 1;
}
if [[ ${ESP_INSTRUCTION_TRACE:-1} == 0 ]]; then
  # Preserve UART and DMA checks while avoiding large passive CPU trace files.
  ln -s /dev/null trace_hart_0.log
  ln -s /dev/null trace_hart_0_commit.log
fi
cat > run.tcl <<EOF
onerror {quit -code 2 -f}
onbreak {quit -code 0 -f}
run $sim_time_ms ms
puts "LCS_SYSTEM_SIM_TIME_LIMIT"
quit -code 3 -f
EOF
vsim -c -suppress 3812,2697,8617,151,143,8386,3009,2685,2718 \
  -L secureip_ver -L unisims_ver -L work -L sqr_bambu \
  -uvmcontrol=disable -t fs +notimingchecks work.testbench work.glbl \
  -G/testbench/soc_inst/gen_mig_model/mig_axiram/kbytes="${ESP_SIM_RAM_KBYTES:-16384}" \
  -do 'do run.tcl' -l simulation.log +gemm_stall="${GEMM_STALL:-0}" \
  +gemm_timeout="$timeout_cycles" \
  > console.log 2>&1
python3 - <<'PY'
from pathlib import Path
import re
log = Path('simulation.log').read_text()
assert 'TEST PASSED' in log and 'TEST FAILED' not in log
assert '** Fatal' not in log and '** Error' not in log
assert 'LCS_SYSTEM_SIM_TIME_LIMIT' not in log
assert '** Failure: Program Completed!' in log, 'missing ESP completion signal'
assert all('Program Completed!' in line for line in log.splitlines()
           if '** Failure:' in line), 'unexpected simulator failure'
assert re.search(r'kernel\s+\d+\s+1\s+', log), 'missing kernel profile'
print('PASS: generated application, numerical check and kernel profiling in full-system ESP')
PY
if ((${#extra_runtime[@]})); then
  python3 "$example/../reusable-gemm/check-system.py" "$generated" simulation.log
fi
printf 'Results: %s\n' "$sim/simulation.log"
