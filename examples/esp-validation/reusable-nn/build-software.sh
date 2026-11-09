#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
app=$(realpath "${1:?Usage: build-software.sh staged-app ESP-root compiled-SoC}")
esp=$(realpath "${2:?ESP root required}")
soc=$(realpath "${3:?Compiled SoC required}")
prefix=${RISCV_PREFIX:-riscv64-unknown-elf-}
dram=${NN_DRAM_BYTES:-16777216}
execution=${NN_EXECUTION:-FPGA}
[[ "$execution" == FPGA || "$execution" == CPU_ONLY ]] || { echo 'NN_EXECUTION must be FPGA or CPU_ONLY' >&2; exit 1; }
[[ "$dram" =~ ^[1-9][0-9]*$ ]] || { echo 'NN_DRAM_BYTES must be positive' >&2; exit 1; }
# The board device tree reserves the first 2 MiB of the upper DRAM half for
# Ethernet. The compact simulation fixture has no such software allocation.
reserved=0
if ((dram >= 1073741824)); then reserved=2097152; fi
cacheline=$(sed -nE 's/.*CFG_CACHE_LINE_SIZE[[:space:]]*: integer := ([0-9]+);/\1/p' "$soc/socgen/esp/esp_global.vhd")
[[ "$cacheline" =~ ^[1-9][0-9]*$ ]] || exit 1
drivers=$esp/soft/common/drivers
cpu=$esp/soft/ariane
libs=$soc/soft-build/ariane/drivers
flags=(-O2 -march=rv64imafdc -mabi=lp64d -mcmodel=medany -fno-common
  -fno-builtin-printf -ffp-contract=off -I"$app" -I"$repo/examples/soda-plugins/include"
  -I"$drivers/baremetal/include" -I"$drivers/common/include"
  -I"$cpu/riscv-tests/env" -I"$cpu/riscv-tests/benchmarks/common"
  -I"$cpu/bootrom" -I"$cpu/riscv-pk/machine" -I"$soc/socgen/esp")
if [[ "$execution" == CPU_ONLY ]]; then
  arena=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["arena_data_bytes"]+256+4096)' "$app/plan.json")
  flags+=(-DSODAP_OP_CPU_ONLY "-DSODAP_OP_CPU_ARENA_BYTES=$arena")
fi
cd "$app"
"${prefix}gcc" "${flags[@]}" -std=gnu99 -c "$here/baremetal_main.c" -o app.o
"${prefix}gcc" "${flags[@]}" -c tensors.S -o tensors.o
"${prefix}gcc" "${flags[@]}" -std=gnu99 -c "$cpu/common/syscalls.c" -o syscalls.o
"${prefix}gcc" "${flags[@]}" -D_init=esp_init_bss -c "$cpu/riscv-tests/benchmarks/common/crt.S" -o crt.o
"${prefix}gcc" "${flags[@]}" -c "$here/../generated-lcs/baremetal-startup.S" -o startup.o
# Rebuild the allocator with this target's actual memory size. Its DMA arena
# begins halfway through DRAM, separate from the code, weights and CPU stack.
"${prefix}gcc" "${flags[@]}" -std=gnu99 -DOVERRIDE_DRAM_SIZE="$dram" \
  "-DDRAM_BASE=(DRAM_BASE_ADDR+$reserved)" -DCACHELINE_WIDTH="$cacheline" \
  -c "$drivers/baremetal/probe/probe.c" -o probe.o
for source in EspRuntime ReusableOperatorRuntime; do
  "${prefix}g++" "${flags[@]}" -std=c++17 -fno-exceptions -fno-rtti \
    -DSODAP_ESP_DEVICE_ID=0x07a '-DSODAP_ESP_DEVICE_NAME="sld,sqr_bambu"' \
    -c "$repo/examples/soda-plugins/lib/sodap/ExecutionEngine/$source.cpp" -o "$source.o"
done
python3 - "$cpu/riscv-tests/benchmarks/common/test.ld" "$dram" "$reserved" "$execution" <<'PY'
from pathlib import Path
import json, sys
s = Path(sys.argv[1]).read_text()
s = s.replace('.text : { *(.text) }', '.text : { *(.text .text.*) }\n  .rodata : { *(.rodata .rodata.*) }\n  /DISCARD/ : { *(.eh_frame .eh_frame.*) }')
s = s.replace('.data : { *(.data) }', '.data : { *(.data .data.*) }')
assert s.count('.sbss : {') == 1 and s.count('.bss : { *(.bss) }') == 1
s = s.replace('.sbss : {', '.sbss : {\n    . = ALIGN(8);\n    __esp_bss_start = .;')
s = s.replace('.bss : { *(.bss) }', '.bss : { *(.bss .bss.* COMMON) . = ALIGN(8); __esp_bss_end = .; }')
dram=int(sys.argv[2])
reserved=int(sys.argv[3])
assert dram & (dram-1) == 0 and 16*1024**2 <= dram <= 2**31
plan=json.loads(Path('plan.json').read_text())
# Even a one-entry scatter/gather table can round the arena to a power-of-two
# chunk. Bound that worst case and leave room for descriptor/probe metadata.
arena=plan['arena_data_bytes']+256
allocation=1 << max(20,(arena-1).bit_length())
assert allocation+65536 < dram//2-reserved, 'DMA arena exceeds target memory'
s += f'\nASSERT(_end + 1048576 < {0x80000000+dram//2}, "Software and stack overlap DMA region")\n'
Path('test.ld').write_text(s)
Path('memory.json').write_text(json.dumps({'dram_bytes':dram,'dma_base':hex(0x80000000+dram//2+reserved),
    'arena_bytes':plan['arena_data_bytes']+256,'execution':sys.argv[4],
    'arena_location':'cacheable BSS' if sys.argv[4]=='CPU_ONLY' else 'uncached DMA'},indent=2)+'\n')
PY
"${LD_LLD:-ld.lld}" -m elf64lriscv -static -T test.ld \
  app.o host.o tensors.o EspRuntime.o ReusableOperatorRuntime.o probe.o syscalls.o crt.o startup.o \
  "$libs/probe/libprobe.a" "$libs/monitors/libmonitors.a" "$libs/utils/baremetal/libutils.a" \
  "$("${prefix}gcc" -march=rv64imafdc -mabi=lp64d -print-file-name=libm.a)" \
  "$("${prefix}gcc" -march=rv64imafdc -mabi=lp64d -print-file-name=libgcc.a)" \
  -o nn_app.exe
"${prefix}readelf" -h nn_app.exe > elf-header.txt
grep -Eq 'Entry point address: +0x80000000' elf-header.txt
"${prefix}nm" --undefined-only nn_app.exe > undefined-symbols.txt
test ! -s undefined-symbols.txt
"${prefix}objcopy" -O srec --gap-fill 0 nn_app.exe ram.srec
"${prefix}objcopy" -O binary nn_app.exe nn_app.bin
sha256sum nn_app.exe nn_app.bin host.o *.bin "$here/baremetal_main.c" \
  "$repo/examples/soda-plugins/lib/sodap/ExecutionEngine/ReusableOperatorRuntime.cpp" \
  > software.sha256
echo "BUILT: $app/nn_app.exe ($dram bytes target DRAM)"
