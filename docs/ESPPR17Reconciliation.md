# PR17 reconciliation and kernel investigation

This local update reconciles Tommaso's PR17 commit
`346b48ef63bea5f7e37f79b12ce5449d518f726c` with the validated MINI campaign.
The campaign is preserved at `e873b2e` on
`local/vc707-mini-validated-20261104`; its original generated files, seven
bitstreams and 14 passing board executions have not been replaced.

The reconciliation is on `local/esp-lcs-reconciled` (`94692d7`). Two additional
MLIR fixes and diagnostic output capture were prepared on
`local/esp-lcs-kernel-investigation`. They are consolidated for review as
`0081c52` on `local/esp-lcs-small-ready`, without the candidate Bambu patch.
These local branches have not been pushed.

This report records the reconciliation checkpoint. The subsequent
[SMALL campaign](ESPSmallCampaign.md) completed full-system RTL validation and
VC707 bitstream generation for ten kernels, including THREEMM, ATAX and TWOMM.
Those new images have not yet been tested on the board.

## Build interface and preserved fixes

Tommaso's naming and stage behavior are retained:

- `--backend esp --stage object` builds both `06_kernel_riscv.o` and `06_verilog.v`.
- Artifacts live under `output/esp/<flow>/`, with `05_hw_acc.ll`,
  `05_kernel_riscv.ll`, `architecture.xml`, and `bambu_syn/`.
- `05_cpu_baseline.ll` and `cpu-baseline/06_kernel_riscv.o` retain the local
  CPU comparison. The validation runners now consume these paths.

The update retains the actual outlined argument order, multiple output checks,
input/output initialization, NaN rejection, profiling, matched host/hardware
DMA slot layout, Bambu device/clock settings, support RTL and memory files,
buffered DMA writes, and explicit baremetal BSS initialization. Generated RTL
and host objects are not patched after generation.

In the configured SODA environment:

```sh
sb-cli init --benchmark_name gemm --dataset SMALL --dtype float32 \
  --backend esp --stage object --device xc7vx485t-2ffg1761 \
  --clock_period 20 --output_dir gemm_small
make -C benches/experiments/gemm_small \
  LCS_PROFILE=true LCS_DMA_WORD_STRIDE=2 all cpu-baseline-object
```

These are the same CLI arguments as Tommaso's flow, with explicit FPGA settings.
The checks used Python/Make in `agostini01/soda-trixie:v19.03` with the locally
built plugin; they did not provision a new Pixi environment. Profiling is
optional. Stride 2 is required by this validation's DMA64 integration: it uses
one float32 value per 64-bit slot, doubling storage and DMA traffic. Default
stride 1 remains dense. Never pair host and RTL built with different settings.

A repeated Make invocation reuses generated artifacts. Toggling profiling
rebuilds both host variants, removes/restores their profiling calls, and leaves
the accelerator LLVM byte-identical.

## Reconciliation checks

- Initial reconciliation: 52/52 lit tests; 145 CLI tests passed, 1 skipped.
- With the local MLIR fixes: 53/53 lit tests; 145 CLI tests passed, 1 skipped.
- All 13 MINI host/CPU pairs compile to RISC-V ELF64 with the double-float ABI
  and expected runtime/libc symbols. The four newly compiling cases also pass
  the normal CLI/Make object-plus-RTL target, not only direct pass invocation.
- All seven original MINI kernels: regenerated host/hardware LLVM, CPU LLVM,
  host objects and test vectors match the validated campaign byte for byte.
- All seven regenerate RTL and pass the backpressured adapter tests, including
  two invocations for each of zero and 50,000-cycle initial write stalls.
  Five RTL sets match after removing the generation-date banner; TRMM and SYRK
  also have generated identifier differences, so byte equality is not claimed.
- The two MLIR fixes leave the LLVM/XML/argument-order artifacts of all nine
  previously compiling MINI kernels unchanged.
- Fresh MINI GEMM full-system simulation passes at `1907 e-9` maximum error
  against the unchanged `1000000 e-9` tolerance, with kernel profiling.

The final MINI full-system rerun also completes the shell runner successfully
(exit 0). An earlier run passed the simulator checks but hit a shell error
after the runner was edited while it was active; that run and its independent
postcheck remain in the local evidence. The final run uses an unchanged runner.

## SMALL status

GEMM, SYR2K, TRMM, SYMM, SYRK, GESUMMV and DOITGEN all generate SMALL host/CPU
objects, pass native CPU numerical checks, and synthesize with Bambu for VC707.
The six non-GEMM SMALL experiments use the default `baseline` flow. GEMM
was checked with both `transformed` and `baseline`, including the normal
`all cpu-baseline-object` targets shown above.
SMALL GEMM also passes repeated-invocation and backpressure adapter tests.
Its full-system ESP simulation passes all 4,200 output values, with maximum
error `11444 e-9` against the unchanged `1000000 e-9` tolerance. The generated
host runs on RISC-V and invokes the generated accelerator through the ESP
socket, NoC and memory tile. The accelerator takes 578,239 cycles at its
boundary, including observed backpressure; the host profiling region includes
start/wait overhead as well and measures 578,718 cycles. The complete profiled
offload takes 5,513,088 cycles, including allocation and data copies. The
simulation runner exits successfully (exit 0).

At this checkpoint there was no SMALL bitstream or SMALL board run. The later
[SMALL campaign](ESPSmallCampaign.md) supplies the per-kernel implementations
and matching host applications; the MINI bitstreams do not execute the larger
shapes.

## Failures and local fixes

| MINI kernel | Original failure | Local result |
|---|---|---|
| THREEMM | Multibuffering rejected intermediate subviews | LLVM, native numerical check, HLS and adapter pass |
| ATAX | Same subview restriction | LLVM, native numerical check, HLS and adapter pass |
| TWOMM | Same subview restriction | LLVM, native numerical check, HLS and adapter pass |
| GEMVER | Subviews, then whole inputs not promoted for DMA | LLVM and HLS now succeed; CPU and RTL fail the original numerical threshold identically |
| BICG | Bambu virtual-dependency analysis exception | Still fails before RTL |
| MVT | Same Bambu exception | Still fails before RTL |

`eefef35` lets intermediate subviews follow the selected ping-pong bank. It
includes their users when finding the common loop and recomputes nested view
types after rebasing. The same rebasing is applied when a compute accumulator
is made private. An executable regression checks static/nested slices through
three iterations, including reuse of bank zero.

`2cfb9eb` stages whole static inputs left by fusion when their uses and subviews
are proven read-only. Unknown users and writes prevent hoisting. This is a
correctness fallback with full-tensor buffers, not an improved tiling strategy
or a guarantee that a larger dataset will fit the FPGA. A regression covers
writes through a subview. A bounded tiled implementation is preferable for
large inputs.

These fixes give 13/13 MINI LLVM generation and 11/13 HLS generation. Ten
kernels pass the adapter numerical checks; GEMVER is the numerical exception.
The three newly passing adapter cases have not yet had their own full-system
or FPGA validation.

### GEMVER numerical result

With the original `1e-3` absolute tolerance, one of 40 outputs fails:

| Quantity at output 32 | Value |
|---|---:|
| Generated CPU and RTL | 4023.986572265625 |
| PyTorch float32 | 4023.98779296875 |
| Float64 calculation using the same float32 input values | 4023.987486571672 |
| Absolute difference from float32 reference | 0.001220703125 |
| Float32 ULP difference | 5 |
| Maximum relative difference | 3.0336e-7 |

All 40 RTL outputs are bit-identical to the generated CPU baseline. This
localizes the observed mismatch to the computation/numerical path shared with
the CPU, rather than an additional ESP DMA or RTL error. Its magnitude is
consistent with different float32 evaluation/accumulation order; this does not
prove arbitrary reassociation is acceptable. The threshold and golden outputs
were not changed. Next compare unfused operation boundaries and accumulation
precision, and agree on an absolute/relative validation policy before changing
acceptance criteria. The failing RTL run stops at its first numerical check;
GEMVER is not counted as passing repeated-invocation/stress validation.

### BICG and MVT

Both still reproduce `absl::container_internal::raw_hash_map<>::at` with the
container Bambu. The earlier debugger evidence remains applicable: a virtual
SSA dependency references a DMA-interface statement owned by a different
function, and `BuildVirtualPhi` looks for its block in the current function.
See [ESPBambuMultiProcessFailure.md](ESPBambuMultiProcessFailure.md).

The candidate Bambu patch remains on the separate investigation branch
`local/esp-lcs-kernel-investigation`; it is not included in this review branch
and is not installed in the campaign compiler. The SMALL campaign uses the
unchanged Bambu binary. BICG/MVT remain excluded until that issue is resolved.

After a matching Bambu rebuild, rerun both cases and verify shared-DMA
arbitration and response ownership with stalls. Eliminating the compiler
exception alone does not prove the generated shared interface works. A single
load/store service or host-sequenced offloads remain alternatives if that
sharing is unsupported.

## Next validation steps

This was the plan at the reconciliation checkpoint. The later SMALL campaign
completed the full-system and FPGA implementation steps below for ten kernels.
Board testing, BICG/MVT and GEMVER remain separate work.

1. Review the two MLIR fixes separately from the build-flow reconciliation.
   They preserve the seven validated MINI designs and expand generation without
   changing the numerical acceptance threshold.
2. Run full-system simulation for THREEMM, ATAX and TWOMM before including them
   in an FPGA campaign. Their current passes cover the generated core and DMA
   adapter, not execution through the RISC-V CPU, NoC and memory tile.
3. For SMALL, complete per-kernel SoC simulation and FPGA implementation before
   claiming board support. Bambu synthesis alone does not establish device fit,
   timing closure or end-to-end correctness for all seven larger designs.
4. Give the Bambu developers the BICG/MVT failing LLVM/XML, debugger evidence
   and candidate patch; validate with a matching compiler before adopting it.
   Investigate GEMVER's floating-point evaluation order independently.

## Evidence

The [result matrix](validation/esp-pr17-20261104/results.csv),
[provenance](validation/esp-pr17-20261104/provenance.json), compiler checksums,
artifact comparisons, UART excerpts and GEMVER diagnosis are kept in
[`validation/esp-pr17-20261104/`](validation/esp-pr17-20261104/).
`NOT_RERUN` distinguishes the preserved earlier SoC results from a fresh run.
The full simulator transcripts are retained locally; their SHA256 fingerprints
are recorded alongside the compact UART extracts.
ESP uses `Failure: Program Completed!` to stop simulation on normal application
completion. The runner requires this exact completion signal, rejects other
simulator failures, and also requires the numerical test and profiling output.

### Larger local artifacts

In `experiments/soda-pr17-reconciled/`:

- `local-results/mini-artifact-comparison.json`, `final-llvm-regression.json`:
  preserved compiler artifacts and regression comparisons.
- `local-results/generation.csv`, `hls.csv`, `small.csv`: build-stage results.
- `local-results/*_mini-adapter/`, `gemm_small-adapter/`: original subset tests.
- `local-results/subviews/{threemm,atax,twomm}/` and
  `{threemm,atax,twomm}_subviews-adapter/`: the first local fix and RTL checks.
- `local-results/gemver-whole/`, `gemver-whole-adapter/`,
  `gemver-hardware-vs-cpu.json`: the numerical failure and output comparison.
- `local-results/bambu-function-scope-proposal/`: candidate patch, isolated build and compatibility-check logs.
- `local-results/reconciliation-evidence/bambu-reproducer/`: preserved dense
  and padded BICG/MVT LLVM/XML, debugger transcripts, binary fingerprint and
  candidate patch, copied from the original campaign for sharing.
- Full-system runs: `/tmp/esp-reconciled-gemm-mini-final/full-system/` and
  `/tmp/esp-reconciled-gemm-small/full-system/`; each retains input checksums,
  ELF, application source inputs and the simulator transcript.

The new commits do not overwrite the old deployment archive or claim that a
new bitstream has been tested on the board.

The subsequent SMALL-only FPGA campaign is described in
[ESPSmallCampaign.md](ESPSmallCampaign.md).
