# Reproducing the ESP runtime and profiling checks

This is a single-GEMM prototype validated against the existing Q16.16 FFN
accelerator on VC707, not validation of newly generated Bambu/dataflow hardware.
The hardware-free checks below are runnable tests, not a newly installed CI job.
The mock checks the host/runtime interface; it does not compute GEMM.

## Compiler and plugin setup

Run from the `soda-benchmarks` repository root inside `agostini01/soda`, with a
sibling `soda-opt` source checkout mounted into the container. LLVM and MLIR are
under `/opt/llvm-project` in the tested image (LLVM 19.1.5). The host remains
Ubuntu; compiler and object-generation commands run inside Docker. The final
ESP baremetal link uses the host's ESP checkout and RISC-V GNU toolchain.

The companion `soda-opt` commit is `19377c0`, on base
`af0f8779949643ae3791f053449509de0a633aa9`. Its
[patch file](patches/0001-Register-MLIR-passes-used-by-the-ESP-IR-marshalling-.patch)
registers two existing MLIR passes; it contains no new lowering implementation.
Apply it only if those registrations are not already present:

```sh
set -euo pipefail
repo="$PWD"
soda_source=$(cd ../soda-opt && pwd)
companion_patch="$repo/docs/patches/0001-Register-MLIR-passes-used-by-the-ESP-IR-marshalling-.patch"
if ! git -C "$soda_source" apply --reverse --check "$companion_patch" 2>/dev/null; then
  git -C "$soda_source" apply --check "$companion_patch"
  git -C "$soda_source" apply "$companion_patch"
fi

cmake -S "$soda_source" -B "$soda_source/build-esp" -G Ninja \
  -DMLIR_DIR=/opt/llvm-project/lib/cmake/mlir \
  -DLLVM_DIR=/opt/llvm-project/lib/cmake/llvm -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXE_LINKER_FLAGS="-Wl,--export-dynamic"
cmake --build "$soda_source/build-esp" --target soda-opt -j 4
cmake -S examples/soda-plugins -B examples/soda-plugins/build -G Ninja \
  -DMLIR_DIR=/opt/llvm-project/lib/cmake/mlir \
  -DLLVM_DIR=/opt/llvm-project/lib/cmake/llvm -DCMAKE_BUILD_TYPE=Release
cmake --build examples/soda-plugins/build --target SODAPlugin mlir_mockesp_runner_utils -j 4

export PATH="$soda_source/build-esp/bin:$repo/scripts:/opt/llvm-project/bin:$PATH"
export PYTHONPATH="/opt/llvm-project/python_packages/mlir_core:/opt/torch-mlir/python_packages/torch_mlir:/opt/soda-opt/python_packages/soda:$repo${PYTHONPATH:+:$PYTHONPATH}"
export SODAP_DIR="$repo/examples/soda-plugins"
export MLIR_PLUGIN_PATH="$SODAP_DIR/build/lib/SODAPlugin.so"
export ESP_RUNTIME_LIB="$SODAP_DIR/build/lib/libmlir_mockesp_runner_utils.so"
export LLVM_TOOLS_DIR=/opt/llvm-project/bin
export LLVM_LIB_DIR=/opt/llvm-project/lib

soda-opt --load-pass-plugin="$MLIR_PLUGIN_PATH" \
  --load-dialect-plugin="$MLIR_PLUGIN_PATH" --help > /dev/null
```

`--export-dynamic` fixes the plugin symbol-resolution failure observed with the
image's original executable. It is a build setting, separate from the source
patch. The two registrations are needed by the fused `esp-ir` recipe, not by
basic runtime/profiling support or conversion-IR emission alone.

## Focused lowering, mock and profiler tests

With the environment above, run:

```sh
bash scripts/test_esp.sh
```

The script uses `set -euo pipefail` and runs the three basic lowering checks
(default, profiling, IR conversion), sequential-offload checks, expected-error
tests, the batch=1 mock under `mlir-cpu-runner`, and the native profiler self-test.
It fails on an unexpected diagnostic, output mismatch, accepted invalid case,
or profiler assertion. It does not need lit installed.

The equivalent tests have `RUN` lines under
`examples/soda-plugins/test/sodap/ESP/`. The `esp-prof.mlir` wrapper makes the C
self-test discoverable by lit. In the tested image, `check-sodap` points to a
missing `build/bin/llvm-lit`; configuring `LLVM_EXTERNAL_LIT` to an installed
lit executable is still necessary to use that target. Passing this script is
not a claim that the entire repository's lit suite or remote CI passed.

The compiler rejects non-f32/non-memref operands and batches that are not
statically known to be 1. IR marshalling additionally rejects dynamic shapes
and more than one offload per function, including nested/different blocks.
This conservative restriction avoids overlapping shared-buffer lifetimes;
runtime marshalling still supports sequential offloads. Reentrant offloads
through calls while a buffer is live remain unsupported: no interprocedural
lifetime analysis or general allocation-failure handling is claimed here.

## Fresh generated GEMM and RISC-V object

Keep using the same container shell and environment. `sb-cli init` only creates
the experiment; the following `make` commands actually build it. A temporary
base directory avoids modifying the repository's experiment registry.

```sh
validation_root=$(mktemp -d /tmp/soda-esp-validation.XXXXXX)

python3 -m sb_cli init --base_dir "$validation_root" --output_dir mock_gemm \
  --benchmark_name gemm --dataset MINI --dtype float32 --flow transformed \
  --backend cpu --stage simulation --instrumentation esp
make -C "$validation_root/experiments/mock_gemm" SCRIPTS_DIR="$repo/scripts"

python3 -m sb_cli init --base_dir "$validation_root" --output_dir esp_review_gemm \
  --benchmark_name gemm --dataset MINI --dtype float32 --flow transformed \
  --backend esp --stage object --instrumentation esp-ir
make -C "$validation_root/experiments/esp_review_gemm" SCRIPTS_DIR="$repo/scripts"

app_dir="$validation_root/experiments/esp_review_gemm/output/esp/transformed/esp-app"
test -s "$app_dir/kernel.o"
test -s "$app_dir/testdata.h"
llvm-readelf -h "$app_dir/kernel.o"
llvm-nm --undefined-only "$app_dir/kernel.o"
sha256sum "$app_dir/kernel.o"
```

The CPU/mock run intentionally prints `TEST FAILED`: the mock does not compute
the accelerator result. Treat it as an integration smoke test, not a numerical
test, and do not use its `make check` as a passing correctness assertion. Use
`--instrumentation esp-ir` in a separate CPU experiment to exercise generated
marshalling loops too; the focused mock-runner test currently executes runtime
marshalling only.

The object must be an ELF64 RISC-V relocatable with the intended hard-float ABI
(`rv64imafdc`, `lp64d` in this setup). Compilation success alone does not resolve
its external functions. Inspect the undefined symbols against the intended
runtime ABI: the tested fused kernel references `esp_*`, `malloc`, and `memcpy`;
unfused variants can also reference `memset`. There is no universal exact
symbol list for every optimization mode. The final baremetal executable,
unlike the relocatable object, must have no undefined symbols.

The previously FPGA-tested fused object has SHA-256
`d5642b4af130fb8b8e1c38c9c7c615ddc897fd416c35b62aaff1259752cc1066`.
Byte equality is a useful reproducibility check with matching tools and input,
not a portable compiler test across LLVM versions.

## ESP dependencies outside this compiler PR

The board environment was not a stock ESP checkout. Its base commit was
`d9c057a609054f8e0f09af1db9a64b17443c05e2`, plus local changes:

- The `ffn_sysc_catapult` accelerator configured for Q16.16, vector length 8,
  with a PLM addressing correction for `K % VEC_LEN != 0`.
- The matching `socs/vc707_ffn` configuration and existing FPGA bitstream.
- Ariane startup clearing `.bss` in `soft/ariane/common/syscalls.c`, enabled by
  `ESP_BAREMETAL_BSS_CLEAR` in `soft/common/drivers/baremetal/common_bare.mk`.
  The latter also selects the default linker script with `-Wl,-dT,...` and
  adds `soft/ariane/common/bss_bounds.ld`, defining the `_fbss` boundary.

The [BSS patch snapshot](patches/esp-ariane-bss-clear.patch) records those three
startup/linker files for reproduction and review. It is not automatically
applied and is not proposed as a universal ESP startup fix. Check applicability
against the ESP revision before using it. The accelerator, SoC and bitstream
are separate artifacts, not supplied by this compiler PR. Record their hashes
when reporting another board run. The runtime's baremetal allocator and the
profiler do not depend on the `.bss` fix: both initialise themselves behind a
magic guard precisely because stock ESP never clears `.bss`. The patch is
recorded because the rest of the board environment (syscalls, the driver's own
statics) was built with it.

To build the staged application, copy `app_dir` out of the container before
removing it. On the host, set `ESP_ROOT`, `ESP_SOC_DIR`, `CROSS_COMPILE`, and
`app_dir` to the matching checkout, SoC directory, toolchain prefix, and copied
application. For the `esp_review_gemm` experiment above:

```sh
set -euo pipefail
app_target="$ESP_ROOT/soft/common/apps/baremetal/esp_review_gemm"
exe="$ESP_SOC_DIR/soft-build/ariane/baremetal/esp_review_gemm.exe"
test ! -e "$app_target"
test ! -e "$exe"
cp -R "$app_dir" "$app_target"
make -C "$ESP_SOC_DIR" esp_review_gemm-baremetal CROSS_COMPILE="$CROSS_COMPILE"
test -s "$exe"
undefined_symbols=$("${CROSS_COMPILE}nm" --undefined-only "$exe")
test -z "$undefined_symbols"
```

Use a fresh application name/output: the outer ESP make target was observed
to return success after a recursive build failure, and an old executable can
otherwise mask failure. Do not use top-level `make -B`: it can regenerate the
SoC configuration from defaults and drop the accelerator mapping. Building
and linking need no simulator; executing needs the FPGA or an appropriate ESP
simulator. FPGA programming is deliberately not part of these instructions.

The recorded VC707 run passed with max error `698090 e-9` below tolerance
`1000000 e-9`; its pack/accelerator/unpack/epilogue measurements were
57,254 / 74,063 / 66 / 61,436 cycles at 50 MHz. These are prior hardware
measurements, not a claim of a new board run after each software edit.
