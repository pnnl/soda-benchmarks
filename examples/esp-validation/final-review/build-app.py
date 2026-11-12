#!/usr/bin/env python3
"""Link generated host objects and runtime against the review fixtures."""
import argparse
import os
import hashlib
import json
from pathlib import Path
import subprocess

tool = Path(__file__).resolve().parent
repo = tool.parents[2]
here = Path(os.environ.get("ESP_REVIEW_ARTIFACTS", repo/"build/final-review")).resolve()
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('kernel')
p.add_argument('--dataset',choices=['MINI','LARGE'],default='MINI')
a = p.parse_args()
root = here / 'generated' / a.dataset / a.kernel
spec = json.loads((root / 'fixture.json').read_text())
esp = here / 'platform/esp'
soc = esp / 'socs/vc707'
out = here / 'apps' / a.dataset / a.kernel
out.mkdir(parents=True, exist_ok=True)
sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
assert spec['dataset']==a.dataset and spec['kernel']==a.kernel
for arg in spec['arguments']:
    assert sha(root / arg['file']) == arg['sha256']
assert sha(root / 'reference.bin') == spec['reference_sha256']
host = root / 'overlap/06_kernel_riscv.o'
order = list(map(int, (root / 'overlap/argument_order.txt').read_text().split()))
assert sorted(order) == list(range(len(spec['arguments']) + 1))
symbols = [f'pb_arg{i}' for i in range(len(spec['arguments']))] + ['pb_output']
header = ['#include <stdint.h>', f'#define PB_NAME "{a.kernel}"',f'#define PB_DATASET "{a.dataset}"',
          f'#define PB_OUTPUT_ELEMENTS {spec["outputs"]}u',
          f'extern const float pb_reference[{spec["outputs"]}];',
          f'extern float pb_output[{spec["outputs"]}];']
assembly = ['.section .rodata', '.balign 128']
for i, arg in enumerate(spec['arguments']):
    header.append(f'extern float pb_arg{i}[{arg["elements"]}];')
    assembly += [f'.global pb_arg{i}', f'pb_arg{i}:',
                 '.incbin ' + json.dumps(str(root / arg['file'])), '.balign 128']
assembly += ['.global pb_reference', 'pb_reference:',
             '.incbin ' + json.dumps(str(root / 'reference.bin')),
             '.section .bss', '.balign 128', '.global pb_output',
             'pb_output:', f'.zero {spec["outputs"] * 4}']
header += ['void forward_kernel(' + ','.join(['float *'] * len(order)) + ');',
           '#define PB_CALL() forward_kernel(' + ','.join(symbols[i] for i in order) + ')']
(out / 'fixture.h').write_text('\n'.join(header) + '\n')
(out / 'tensors.S').write_text('\n'.join(assembly) + '\n')
drivers = esp / 'soft/common/drivers'
cpu = esp / 'soft/ariane'
libs = here / 'platform/drivers'
plugin = repo / 'examples/soda-plugins'
prefix = '/opt/riscv/bin/riscv64-unknown-elf-'
flags = ['-O2', '-march=rv64imafdc', '-mabi=lp64d', '-mcmodel=medany',
         '-fno-common', '-fno-builtin-printf', '-ffp-contract=off']
for path in [out, plugin / 'include/sodap/ExecutionEngine',
             drivers / 'baremetal/include', drivers / 'common/include',
             cpu / 'riscv-tests/env', cpu / 'riscv-tests/benchmarks/common',
             cpu / 'bootrom', cpu / 'riscv-pk/machine', soc / 'socgen/esp']:
    flags += ['-I' + str(path)]
commands = []
def run(args):
    args = list(map(str, args)); commands.append(args)
    subprocess.run(args, cwd=out, check=True)
dram=16*1024**2 if a.dataset=='MINI' else 1024**3
reserved=0 if a.dataset=='MINI' else 2*1024**2
sources = [(repo / 'scripts/lib/memref_copy.c', 'memref_copy.o', []), (tool / 'baremetal.c', 'app.o', ['-DPB_SIMULATION'] if a.dataset=='MINI' else []),
           (out / 'tensors.S', 'tensors.o', []),
           (plugin / 'lib/sodap/ExecutionEngine/esp_prof.c', 'profiler.o', []),
           (cpu / 'common/syscalls.c', 'syscalls.o', ['-fno-builtin']),
           (cpu / 'riscv-tests/benchmarks/common/crt.S', 'crt.o', ['-D_init=esp_init_bss']),
           (repo / 'examples/esp-validation/generated-lcs/baremetal-startup.S', 'startup.o', []),
           (drivers / 'baremetal/probe/probe.c', 'probe.o',
            [f'-DOVERRIDE_DRAM_SIZE={dram}', f'-DDRAM_BASE=(DRAM_BASE_ADDR+{reserved})', '-DCACHELINE_WIDTH=128', '-Dprobe=pb_driver_probe'])]
for src, obj, extra in sources:
    run([prefix + 'gcc', *flags, *extra, '-c', src, '-o', obj])
runtime = plugin / 'lib/sodap/ExecutionEngine/EspRuntime.cpp'
run([prefix + 'g++', *flags, '-std=c++17', '-fno-exceptions', '-fno-rtti',
     '-DSODAP_ESP_DEVICE_ID=0x07a', '-DSODAP_ESP_DEVICE_NAME="sld,sqr_bambu"',
     '-I' + str(plugin / 'include'), '-c', runtime, '-o', 'runtime.o'])
s = (cpu / 'riscv-tests/benchmarks/common/test.ld').read_text()
s = s.replace('.text : { *(.text) }', '.text : { *(.text .text.*) }\n  .rodata : { *(.rodata .rodata.*) }\n  /DISCARD/ : { *(.eh_frame .eh_frame.*) }')
s = s.replace('.data : { *(.data) }', '.data : { *(.data .data.*) }')
s = s.replace('.sbss : {', '.sbss : {\n    . = ALIGN(8);\n    __esp_bss_start = .;')
s = s.replace('.bss : { *(.bss) }', '.bss : { *(.bss .bss.* COMMON) . = ALIGN(8); __esp_bss_end = .; }')
s += f'\nASSERT(_end + 1048576 < {0x80000000+dram//2}, "CPU image overlaps DMA arena")\n'
(out / 'test.ld').write_text(s)
libm = Path(subprocess.check_output([prefix + 'gcc', '-march=rv64imafdc', '-mabi=lp64d', '-print-file-name=libm.a'], text=True).strip())
libgcc = Path(subprocess.check_output([prefix + 'gcc', '-march=rv64imafdc', '-mabi=lp64d', '-print-file-name=libgcc.a'], text=True).strip())
archives = [libs / 'probe/libprobe.a', libs / 'monitors/libmonitors.a', libs / 'utils/baremetal/libutils.a', libm, libgcc]
run(['ld.lld-16', '-m', 'elf64lriscv', '-static', '-T', 'test.ld',
     *[obj for _, obj, _ in sources], 'runtime.o', host, *archives, '-o', 'app.exe'])
assert '0x80000000' in subprocess.check_output([prefix + 'readelf', '-h', out / 'app.exe'], text=True)
assert not subprocess.check_output([prefix + 'nm', '--undefined-only', out / 'app.exe']).strip()
run([prefix + 'objcopy', '-O', 'binary', 'app.exe', 'app.bin'])
if a.dataset=='MINI':run([prefix+'objcopy','-O','srec','--gap-fill','0','app.exe','ram.srec'])
record = {'kernel': a.kernel, 'dataset': a.dataset, 'mode': 'esp', 'variant': 'integration_thin_candidate',
          'status': 'LINKED_NOT_EXECUTED', 'dram_bytes': dram, 'dma_base': hex(0x80000000+dram//2+reserved),
          'simulation_uart_divider': 1 if a.dataset=='MINI' else None,
          'guard_check': False, 'timing_scope': 'Raw total and total minus measured device discovery; allocation includes configuration',
          'fixture_sha256': sha(root / 'fixture.json'), 'host_object_sha256': sha(host),
          'elf_sha256': sha(out / 'app.exe'), 'binary_sha256': sha(out / 'app.bin'),
          'source_sha256': {str(src): sha(src) for src in [*[s[0] for s in sources], runtime, out / 'fixture.h', out / 'test.ld']},
          'library_sha256': {str(f): sha(f) for f in archives}, 'commands': commands,
          'compiler': subprocess.check_output([prefix + 'gcc', '--version'], text=True).splitlines()[0]}
(out / 'build.json').write_text(json.dumps(record, indent=2) + '\n')
print(out / 'app.exe')
