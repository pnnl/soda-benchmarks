# Reusing one GEMM accelerator

`gemmreuse` follows the existing PolyBench Python catalog, dataset, experiment,
test-data, and numerical-check interfaces. It is an additional validation app,
not an upstream PolyBench kernel. It computes `H = A @ B; Y = H @ W` and returns
both tensors so that validation checks the intermediate as well as the result.

For MINI, the two offloads are:

| Invocation | A | B | C | M, N, K | Instance |
| --- | --- | --- | --- | --- | --- |
| 0 | A: 7×13 | B: 13×9 | H: 7×9 | 7, 9, 13 | 0 |
| 1 | H: 7×9 | W: 9×5 | Y: 7×5 | 7, 5, 9 | 0 |

H stays at arena byte offset 1792 between calls. Inputs A/B/W are copied once,
both calls complete sequentially, and H/Y are copied out for validation after
the second call. No hardware regeneration, reset, or host copy of H occurs
between the offloads. The default hardware has local 4×4×4 tiles; both calls
exercise partial tiles. The arena occupies 2936 bytes including its header.

## Implementation

The [design contract](../../../docs/ESPReusableOperators.md) describes the
separation between invocation semantics, implementations, and instance binding.

* `sodap-reusable-gemm-to-esp` isolates supported matmuls into runtime calls with
  `sodap.operator`, `sodap.invocation`, and `sodap.instance` attributes. It has no
  tile parameters. Dimensions remain visible constants in `host.mlir`.
* `sodap-generate-reusable-gemm` generates hardware from an empty module, a
  schedule and three local tile extents. M/N/K, addresses, row strides, alpha and beta arrive
  in a versioned DMA descriptor each launch. `interface.h` is a copy of the
  shared compiler/runtime ABI header.
* `esp_gemm_open/copy_in/invoke/copy_out/close` own an explicit context and shared
  arena. `invoke` validates bounds, alignment and output aliasing, encodes the
  descriptor, starts the device, waits for completion, then acknowledges it.
* The existing DMA64 adapter drains writes before signaling ESP completion.
  A float32 occupies the low word of an eight-byte slot; padding is zero.

The current compiler accepts static contiguous f32 matmul graphs, including
batch-one matmuls, with zero-initialized outputs. The runtime/hardware also
support row strides and general alpha/beta, tested separately. Dynamic Python
shapes, fused epilogues, transposition, general batching and asynchronous host
calls are not implemented in this path.
The current ESP baremetal allocator does not physically reclaim memory on
close; one arena is allocated for the entire graph, not once per invocation.

## Build the application and hardware

Run in the SODA environment with the plugin built and `MLIR_PLUGIN_PATH` and
`SODAP_DIR` set (see [ESPValidation.md](../../../docs/ESPValidation.md)). The
local run uses `agostini01/soda-trixie:v19.03`, LLVM 19, and PandA 2026.06.
Commands below run from the repository root:

```bash
python3 -m sb_cli init --base_dir "$PWD/build/experiments" \
  --output_dir gemmreuse --benchmark_name gemmreuse --dataset MINI \
  --dtype float32 --backend esp --stage llvm \
  --device xc7vx485t-2ffg1761 --clock_period 20
exp=build/experiments/experiments/gemmreuse
make -C "$exp" LCS_REUSABLE_GEMM=true LCS_DMA_WORD_STRIDE=2 \
  LCS_PROFILE=true object cpu-baseline-object
generated="$exp/output/esp/baseline"
```

Inspect `host.mlir` for the two attributed calls, `accelerator.mlir` for the
parameterized hardware, `schedule.txt` for local tile settings, and
`hardware-inputs.sha256` / `bambu_syn/rtl.sha256` for artifact identities.
`esp-app/` contains the staged application and runtime sources;
`06_kernel_riscv.o` contains the generated host code.

Set `LCS_GEMM_SCHEDULE=lcs` to generate concurrent load, compute and store
tasks. The default remains `sequential`. The LCS implementation uses the common
Dataflow ping-pong transformation to allocate two slots for each shared tile;
tokens retain a slot until its consumer finishes. Each stage receives the same
runtime descriptor, and partial tiles remain runtime decisions. The descriptor,
host LLVM and runtime ABI are unchanged by this scheduling choice.

The generator marks whole tile transactions with `sodap.dataflow_tile` on
`scf.execute_region`. Ping-pong runs before canonicalization so ownership spans
all scalar accesses within the tile, rather than one scalar iteration.
See [the integration report](../../../docs/ESPReuseLCSIntegration.md) for validation
and the distinction between component speedup and total application time.

For FPGA logic synthesis (Vivado 2023.2), use a fresh output directory:

```bash
vivado -mode batch -source examples/esp-validation/reusable-gemm/synthesize.tcl \
  -tclargs "$generated" build/fpga-synthesis
```

This synthesizes the accelerator plus DMA adapter for `xc7vx485tffg1761-2` and
produces a checkpoint, utilization, and timing estimates. It is out-of-context
synthesis, not a placed/routed SoC or an FPGA bitstream.

## Native and component RTL validation

Run the native check inside the compiler environment and the adapter check
where Icarus Verilog is installed. Both must see the same output and fixture
directories:

```bash
bash examples/esp-validation/reusable-gemm/run-native.sh "$generated" build/adapter
bash examples/esp-validation/reusable-gemm/run-adapter.sh "$generated" build/adapter
```

The native test executes the actual generated host LLVM against the reference
implementation of the runtime ABI. Optional trace capture records each
descriptor, initial arena, expected result and write mask. RTL simulation
loads the initial payload once; between calls it changes only the descriptor.
Thus H consumed by call 1 must be produced by RTL call 0. Each call checks
numerical outputs and exact preservation of every non-output word, including
inputs, stride gaps and padding. The runner tests normal DMA and a 50,000-cycle
write-grant stall. No reset occurs between calls.

The separate ten-call suite exercises coefficients, padded row strides,
dimensions smaller than tiles, and repeated accumulation:

```bash
mkdir -p build/adapter-ten
c++ -std=c++17 -O2 -ffp-contract=off -DSODAP_GEMM_TRACE \
  -I examples/soda-plugins/include \
  examples/soda-plugins/test/sodap/ESP/Inputs/test_reusable_gemm.cpp \
  examples/soda-plugins/lib/sodap/ExecutionEngine/ReusableGemmRuntime.cpp \
  -o build/native-ten
SODAP_GEMM_TRACE_DIR="$PWD/build/adapter-ten" build/native-ten
bash examples/esp-validation/reusable-gemm/run-adapter.sh "$generated" build/adapter-ten
```

To test another hardware schedule, generate into a distinct experiment/output
directory with `LCS_GEMM_TILE_M=2 LCS_GEMM_TILE_N=8 LCS_GEMM_TILE_K=3`. Compare
`05_kernel_riscv.ll` byte-for-byte, and run both implementations against the
same fixtures. Tile changes must affect hardware while preserving host calls.

To check recovery from rejected descriptors without resetting the accelerator:

```bash
python3 examples/esp-validation/reusable-gemm/prepend-invalid-descriptors.py \
  build/adapter build/adapter-invalid
bash examples/esp-validation/reusable-gemm/run-adapter.sh "$generated" build/adapter-invalid
```

## Full-system RTL validation

Use the existing compiled ESP VC707 fixture with Ariane, the compatible
`sqr_bambu` socket, DMA64, and disabled L2/LLC. Run from a host with the RISC-V
compiler, ModelSim license, and `ld.lld-16` available:

```bash
export LD_LLD=ld.lld-16 ESP_INSTRUCTION_TRACE=0
export ESP_SIM_RAM_KBYTES=16384 ESP_SIM_TIME_MS=100
export LCS_TIMEOUT_CYCLES=2000000
bash examples/esp-validation/generated-lcs/run-system.sh \
  "$generated" "$ESP_ROOT" "$COMPILED_SOC" build/system-run
```

The runner rebuilds the accelerator simulation library, compiles and links
the complete RISC-V software, checks its ELF entry point and unresolved symbols,
and boots that binary in the full SoC. The expected final signals are numerical
`TEST PASSED`, kernel profiling, and ESP's `Program Completed!`. The additional
`check-system.py` compares generated call dimensions, addresses and bindings
with CPU logs and physical RTL start/completion counts, producing
`reusable-validation.json`.

If vendor simulation libraries are unavailable, `prepare-sim-libraries.sh`
builds the actual Vivado models needed by this fixture. It uses the full vendor
component package including legacy declarations. Set `ESP_MODELSIM_INI` to its
generated `modelsim.ini`, `ESP_MODELSIM_WORK` to a compatible compiled work
library, and `ESP_RECOMPILE_SOC_TOP=1` to refresh the dependent ESP wrappers.
The preparation helper is specific to this fixture, not a complete Vivado
simulation-library installation. Always choose a fresh system run directory.

## Local results

The current run and artifact paths are recorded in
[ESPReusableGemmValidation.md](../../../docs/ESPReusableGemmValidation.md).
