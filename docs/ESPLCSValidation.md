# Local validation of generated load/compute/store accelerators

This local experiment starts from PR #17 at
`e5d2515f876a76c8e27a88886019017417caacd0`. Its scope is a fixed-shape,
float32 PolyBench MINI kernel with its own generated accelerator and matching
host application. Runtime-configurable accelerators are outside this scope.
The subsequent seven-kernel full-system simulation and VC707 image campaign
is documented in [ESPVC707Campaign.md](ESPVC707Campaign.md). The initial
GEMM measurements below are retained as historical evidence.

For the later reconciliation with Tommaso's `346b48e` update, SMALL checks,
and fixes for intermediate subviews, see
[ESPPR17Reconciliation.md](ESPPR17Reconciliation.md). Results below describe
the earlier checkpoint.

## Changes under test

- Connect the ESP `object` stage to the generated LCS host, and add a `verilog`
  target consuming its matching LLVM and architecture XML.
- Preserve the outliner's actual argument order, handle multiple returned
  tensors in the generated test application, and reject NaNs in validation.
- Copy all arguments into shared memory before starting the accelerator,
  including arguments that are read and written or only partially overwritten.
- Add opt-in whole-kernel profiling and a CPU baseline generated from the same
  outlined kernel. The offload also reports allocation, packing, start/wait and
  unpacking separately.
- Reject unsupported direct argument-to-argument DMA copies with a diagnostic,
  and avoid erasing a shared subview twice.
- Add an optional physical layout with one f32 per 64-bit DMA slot. This makes
  odd-length and odd-offset transfers compatible with the existing ESP adapter
  without editing generated RTL. It doubles tensor storage and traffic; dense
  layout remains the default.

The build commands, layout contract and profiling region definitions are in
[ESPBackend.md](ESPBackend.md#generated-loadcomputestore-accelerators).

## Validation levels and results

Compiler/plugin and mock execution tests: **52/52 lit tests passed**.
CLI tests: **145 passed, 1 skipped**. The mock test runs the lowered host with
both dense and padded layouts, preserves partially overwritten inputs across
two invocations, and checks profiling counts and nesting.

The following campaign used `agostini01/soda-trixie:v19.03`, float32 MINI,
`LCS_PROFILE=true`, `LCS_DMA_WORD_STRIDE=2`, and Bambu target
`xc7vx485t-2ffg1761` with a 20 ns clock constraint. Bambu reports
`PandA 2026.06 - Revision unknown-unknown`; generated directories record its
version output, exact command, and input/RTL checksums.

| Kernel | Host + CPU RISC-V objects | Native CPU numerical check | Bambu RTL | Adapter RTL numerical check |
|---|---|---|---|---|
| GEMM | Pass | Pass | Pass | Pass |
| SYR2K | Pass | Pass | Pass | Pass |
| TRMM | Pass | Pass | Pass | Pass |
| SYMM | Pass | Pass | Pass | Pass |
| SYRK | Pass | Pass | Pass | Pass |
| BICG | Pass | Pass | Fail | Not run |
| GESUMMV | Pass | Pass | Pass | Pass |
| DOITGEN | Pass | Pass | Pass | Pass |
| MVT | Pass | Pass | Fail | Not run |

Each adapter test runs the unedited generated IP, its support RTL and the
32-to-64-bit adapter against a backpressured memory model. It checks every
output against PyTorch at the unchanged absolute tolerance of `1e-3`, and
checks that inputs and padding remain unchanged. Each test performs two
invocations without resetting the IP and includes a 50,000-cycle write-control
stall. These are component RTL tests, not complete ESP SoC simulations, and
their cycle counts are stress-test durations rather than performance results.

The native CPU baseline also passes for all nine kernels in dense layout.
The baseline is the CPU implementation of the same tiled/fused outlined
kernel, not a tuned matrix library. Native timings use nanoseconds and cannot
establish an ESP accelerator speedup.

## Initial full-system GEMM comparison

GEMM also passed full-system RTL simulation in both modes: the generated host
and accelerator, and the generated CPU baseline. These runs include the RISC-V
processor, ESP socket, NoC and memory tile. L2/LLC were disabled; the DMA port
was 64 bits. No generated RTL or host object was hand-edited. This was the
first full-system run, before expanding to the other six successful HLS kernels
and adding write buffering to the adapter. Use the subsequent campaign report
for the current implementation and measurements.

Both applications reported `max error = 1907 e-9`, below the unchanged
`1000000 e-9` threshold, and one call in the kernel profiling region.

| Region | CPU baseline (cycles) | ESP offload (cycles) |
|---|---:|---:|
| Complete kernel / offload (7) | 403,082 | 1,065,415 |
| Outer application timer (0) | 403,707 | 1,065,994 |
| Allocation and setup (5) | — | 540,251 |
| Pack (1) | — | 340,605 |
| Runtime start/wait (2) | — | 59,623 |
| Unpack (3) | — | 123,577 |

The RTL adapter independently measured 59,144 cycles from accelerator start to
completion, with 1,302 read and 100 write requests. No artificial stall was
enabled in these full-system runs.

These are single cold invocations. Allocation includes first-use device/cache
probing, shared-buffer/page-table setup and initialization. The whole offload
is approximately **2.64 times slower** than this CPU baseline despite the
shorter hardware execution interval. Runtime setup and copying dominate this
small workload, and the padded layout doubles storage/traffic. These results
establish functionality and show where time goes; they are not steady-state or
FPGA performance results. Reuse of allocations, efficient copying and dense
DMA transfers need separate evaluation before making a speedup claim.

ESP's testbench ends a successful application with the expected simulator
message `** Failure: Program Completed!`. The runner requires that completion
signal, `TEST PASSED`, and the kernel profile, and rejects other errors/failures.

## Unresolved cases

BICG and MVT fail inside Bambu with:

```text
absl::container_internal::raw_hash_map<>::at
Please report bugs to <panda-info@polimi.it>
```

Both fail in dense and padded layouts. Exception traces for both localize
the failing map lookup to Bambu's `BuildVirtualPhi::InternalExec()`, reached
through `FunctionFrontendFlowStep::Exec()`. This is a frontend failure, before
RTL generation. Subsequent debugging found that the analysis follows a
DMA-interface dependency into a different generated process, then looks up
that process's basic block in the current function. The evidence, candidate
correction and reproducible inputs are described in
[ESPBambuMultiProcessFailure.md](ESPBambuMultiProcessFailure.md).
Successful LLVM/object generation is not successful HLS.

The original review also found MINI THREEMM, GEMVER, ATAX and TWOMM failing
during multibuffering. This experiment concentrates on the nine kernels that
already generate LLVM; it does not fix those four transformation failures or
claim support for all datasets.

Efficient dense partial writes are still an integration task. Rounding a
five-word write up to six words would overwrite adjacent storage. The optional
padded layout is a conservative functional path until that is addressed.

## Reproducing a candidate and its baseline

After configuring the compiler/plugin environment described in
[ESPValidation.md](ESPValidation.md):

```sh
sb-cli init --benchmark_name gemm --dataset MINI --dtype float32 \
  --flow transformed --backend esp --stage object \
  --device xc7vx485t-2ffg1761 --clock_period 20 --output_dir gemm_lcs
make -C benches/experiments/gemm_lcs \
  LCS_PROFILE=true LCS_DMA_WORD_STRIDE=2 object cpu-baseline-object verilog
examples/esp-validation/generated-lcs/run-adapter.sh \
  benches/experiments/gemm_lcs/output/lcs /tmp/gemm_lcs_adapter
```

Replace `gemm` with another benchmark name and use a separate output directory.
Keep host, RTL and `layout.txt` from the same build together. Changing the
profiling or layout options invalidates the corresponding generated artifacts.

For full-system validation, `generated-lcs/run-system.sh` reuses the existing
compiled ESP validation SoC with L2/LLC disabled and replaces its accelerator
library and baremetal program. It requires ModelSim, ESP's RISC-V libraries/toolchain, and
LLD to link Clang 19 objects. It checks for numerical success and a kernel
profiling record, and records input checksums.

This fixture uses the existing `sld,sqr_bambu` socket (device ID `0x07a`). The
patch does not automate ESP tile/SoC generation or create FPGA bitstreams.

```sh
examples/esp-validation/generated-lcs/run-system.sh \
  benches/experiments/gemm_lcs/output/lcs "$ESP_ROOT" "$ESP_COMPILED_SOC" \
  /tmp/gemm_lcs_system
LCS_SYSTEM_MODE=cpu-baseline \
  examples/esp-validation/generated-lcs/run-system.sh \
  benches/experiments/gemm_lcs/output/lcs "$ESP_ROOT" "$ESP_COMPILED_SOC" \
  /tmp/gemm_lcs_cpu_system
```

Use fresh run directories. Both programs must use the same processor, clock,
cache settings, input tensors and numerical tolerance. Compare the `kernel`
region (7), or the outer `total` region (0). The offload's kernel region includes
allocation, packing, runtime launch/wait, unpacking and cleanup. Comparing only
`accel` (2) would omit these costs. Artificial DMA stalls are disabled in this
system runner unless explicitly requested with `GEMM_STALL`.
