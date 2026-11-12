# Native ESP integration candidate — 12 November 2026

This branch combines Tommaso's `feature/esp-lcs` at
`3522079d022797092ba9db4c588ef755a5a04ae9` with the locally validated DMA and
reuse changes. It is a candidate for review, not the revision measured on FPGA
in experiment 1. No ESP socket RTL is changed. New FPGA measurements wait until
we agree on this compiler and its adapter.

## What remains from Tommaso's revision

The TOSA transpose-raising/folding and explicit nested hardware module remain.
The host retains the boundary transposes and an explicit `dataflow.launch`.
The host and accelerator have separate lowering paths. Explicit tile-size
overrides and the divisor-based DMA cost model remain available.

We use that model to propose tilings, then score the *scheduled* IR. The original
estimate precedes demand-driven scheduling and cannot see intermediate
recomputation or the final ping-pong allocations. This distinction matters for
ATAX and chained matrix products.

## Changes to review

| Commit / files | Behavior and reason |
| --- | --- |
| `e791f47`: `OutstandingLoad.cpp`, `PingPong.cpp`, `ConvertDataflowToLLVM.cpp`, stream operations | A read is admitted by its receiver, and publication is ordered. Accepting a descriptor must not leave the socket waiting for a consumer that needs another DMA transaction first. |
| `54d13b4`: `Schedule.cpp`, `Reuse.h`, `PingPong.cpp` | Infer retained intermediate slabs from static producer/consumer accesses, within a byte budget. Unsupported or oversized patterns keep the existing demand-driven schedule. |
| `5449e3a`: `CopyToDMA.cpp`, `ConvertDataflowToLLVM.cpp` | Wait for a retained, completed output tile before publishing its write descriptor. Preserve the token until the payload is consumed. The ordered peek lowers to a synthesizable `nb_peek` retry loop. |
| `41b893b`: `Schedule.cpp`, `PingPong.cpp`, pass options | Non-overlap is the same tiled/reuse schedule serialized into one task. It does not allocate entire LARGE tensors locally. |
| `362f90d`: PolyBench `trmm.py` | Use canonical unit-diagonal TRMM semantics. The SPECTRA baseline and reference fixtures use this correction too. |
| `b5919d4`: reusable operator generator, ABI/runtime, graph lowering, EELS frontend | Preserve runtime shape/address programming; add Conv1D extraction, fused LeakyReLU and multiple outputs. |
| `04f86bb`: `SODAPTransforms.cpp`, `ESPPasses.cpp` | Diagnose all-zero parallel tiling and incompatible repeated fixed-kernel launches instead of crashing or silently using the first call's allocation. |
| `c31b8e4`: `ReportLCSCost.cpp`, `Schedule.cpp`, `scripts/lcs/select_tiles.py`, `tosa_to_lcs.sh` | Report scheduled external DMA traffic, arithmetic iteration counts and logical local storage; rank a bounded set of proposals and enforce the total storage cap. Emit retained/over-budget decisions. |
| `70a4c0d`: `examples/esp-validation/native-dma/`, generated-LCS deployment helpers | Add the thin adapter and compare it with the buffered adapter using identical generated IP and vectors. Check HLS input provenance before selecting thin. |
| `3162714` and follow-up documentation | Portable full-system replay, stale-artifact checks and the review payload. |

All paths in the first eight rows are relative to
`examples/soda-plugins/lib/sodap/` unless otherwise indicated. Tests live beside
the corresponding pass tests. Review the commits individually rather than
treating the branch as one monolithic patch.

## DMA contract and the thin adapter

Bambu exposes separate read-request, read-data, write-request and write-data
streams. These are adapted to native ESP DMA; they are not a full AXI master.
Separate channels do not guarantee that the native socket can complete an
arbitrary cyclic dependency between requests and payloads. A descriptor can
occupy a shared socket resource while the accelerator waits for another
transaction to produce or consume that descriptor's payload.

The compiler therefore admits reads from the receiving process and publishes a
write only after its output tile is ready and retained. The thin adapter removes
the write-chunk payload buffer and its admission policy. It still needs a read
beat register, a write packing register, 32/64-bit conversion, descriptor unit
conversion, ready/valid handling and completion tracking. Those are interface
adaptation, not a large intermediate cache.

The current thin write path requires aligned, even-word transactions. The LCS
flow defaults to `LCS_DMA_WORD_STRIDE=2`: one FP32 value plus one padding word per
64-bit DMA beat. Dense stride-1 output is not supported by this thin adapter.
`native_dma_contract.json` binds the pipeline's LLVM, architecture and layout
files to the HLS inputs. It prevents accidentally selecting thin with stale
compiler output; it is not a formal proof for arbitrary modified RTL.

The buffered adapter remains available with `--wrapper buffered` in the Python
deployment tools or `LCS_WRAPPER=buffered` in `generated-lcs/run-system.sh`.
Legacy generated-GEMM and reusable-network scripts keep their existing adapter.
Their custom operator-generation path is not silently switched to thin.

Fixed-kernel `dataflow.launch` and reusable operator dispatch remain distinct:
the former requires compatible static operand types on repeated calls; the
reusable GEMM/conv/add path programs dimensions, tensor offsets and operation
flags at each offload, while numerical types and hardware resources stay fixed.
This branch preserves that runtime/ABI separation.

Thin does **not** imply higher throughput. For the same MINI overlapping IP,
the first component invocation took 11,220 vs 11,180 cycles for ATAX and
29,214 vs 28,766 for GEMM (thin vs buffered); 3MM took 24,464 vs 24,537.
These are deterministic component tests, not FPGA results. The purpose is to
place the DMA dependency contract in the compiler and simplify integration.

## Reuse allocation: 2MM versus 3MM

The inference currently recognizes a static, regular, single-producer,
single-consumer edge. It identifies the slab reused across iterations, proves
the supported coverage/dependence pattern, and admits it under a conservative
three-copy byte reservation. Ping-pong lowering then implements the retained
storage. It does not use a kernel name, shape-specific constant, or hand-written
MLIR annotation. Fanout, dynamic indexing and arbitrary access patterns are not
claimed to be covered.

Current defaults are a 256 KiB reuse admission budget and a **320 KiB total
logical storage cap** after scheduling. The reuse budget is within the total,
not additional to it. Logical bytes are not a prediction of BRAM blocks or a
guarantee of place-and-route feasibility.

With the candidate's automatically selected LARGE tilings:

| Kernel | Reuse decision | Scheduled matrix-product iterations |
| --- | --- | --- |
| 2MM | Retain `[20,900]`; conservative reservation 216,000 bytes; final logical storage 161,466 bytes | 792,000,000 + 864,000,000: each product evaluated once |
| 3MM | `[40,900]` needs 432,000 bytes; `[900,1100]` needs 11,880,000; both exceed the reuse budget | 7,920,000,000 + 23,760,000,000 + 792,000,000 |

The 3MM minimum for these shapes is 2.7 billion matrix-product iterations; this
candidate schedules 32.472 billion. This remains a reason to evaluate three
offloads on one programmable GEMM engine with DRAM intermediates. It is an
alternative execution strategy, not an automatic partitioning decision already
implemented by this pass.

The historical FPGA revision used different tiles: it retained a smaller first
3MM intermediate but repeatedly recomputed the other. Do not attribute its FPGA
latency to the candidate's new tiling. `reuse_decisions.txt`, `tile_search.json`
and `schedule_cost.json` make these decisions inspectable.

## Tiling policy and limits

The bounded search considers Tommaso's divisor-model proposals at five storage
caps plus the historical generic maximum tile size of 10. Every proposal is
scheduled and checked against the total cap. Rank by scheduled DMA cost, then
arithmetic iterations, then logical storage. Both LCS modes use the same overlap
selection policy and conservative reuse admission, so the ablation does not
silently change tiles or recomputation.

Defaults: 64 cost units per row request, 1 per 32-bit DMA word. These are
heuristic weights, not calibrated VC707 latency measurements. The search is
neither exhaustive nor globally optimal, and does not select banking, unrolling
or HLS pipeline directives. The reduction-dimension model and physical resource
model remain future optimization work.

## Validation and preserved experiment 1

See `ESPFinalIntegrationResults.json` for the recorded candidate results.
The qualification covers all ten kernels: GEMM, ATAX, 2MM, 3MM, SYRK, SYR2K,
SYMM, TRMM, DOITGEN and GESUMMV.

- 81 compiler tests and 13 reusable graph/frontend tests.
- Both modes lower for MINI and LARGE: 40 complete compiler outputs.
- Actual generated host LLVM plus the serialized scheduled tile matches the
  independent MINI reference for all ten kernels, including host transposes.
- 20 fresh MINI Bambu HLS builds, targeting `xc7vx485t-2ffg1761`, 20 ns.
- Both wrappers × both modes × ten kernels pass component RTL. Each case runs
  twice without reset at each of two initial write stalls (0 and 50,000 cycles):
  160 checked invocations in total. The model also applies ordinary backpressure.
- Full-system native 2x2 ESP results and the relocated UID test are recorded in
  the results file. LARGE HLS, FPGA implementation and measurements are deferred.
- Fresh reusable conv/add/GEMM RTL passes the EELS encoder, a synthetic Conv1D
  multi-output fixture and a small ResNet-style residual graph with stalled DMA.
  This does not constitute a new full ResNet-18 FPGA validation.

Existing experiment-1 results remain at their original paths on emilia. The
campaign directory `experiments/esp-final-integration-20261112/reference` stores
the source archives, copied light records and a manifest covering 4,024 artifact
hashes and all 20 measured overlap/non-overlap records. Large bitstreams remain
in place and are hashed, rather than duplicated. SPECTRA/CPU records retain
their original provenance. New candidate results must use a distinct campaign.

## Reproduce from Tommaso's checkout

```bash
cd /home/tommaso/esp-final-review-20261112/compiler
source examples/esp-validation/final-review/env.sh
export SODA_IMAGE=sha256:192a841f6f5cf6d15103725d4cb4c2ed88b275b4ee334a53f158bc9c3aa33ec4
```

The local payload is `build/final-review`; the licensed simulator/platform
libraries and generated hardware are local artifacts, not files in this PR.
Previous worktrees are preserved. The following replay uses the new accelerator,
new generated host and the unchanged compiled ESP platform:

```bash
python3 examples/esp-validation/final-review/build-app.py atax
python3 examples/esp-validation/final-review/run-system.py atax --run atax-review-1
```

Use a new `--run` name each time. Add `--variant nonoverlap` to replay the
serialized IP, or `--wrapper buffered` for a controlled adapter comparison.
The default is overlapping LCS with thin. The software checks all output values;
logs, source hashes and accelerator cycles go into `build/final-review/systems`.

For a quick component test without ModelSim or a board:

```bash
python3 examples/esp-validation/native-dma/run-component.py \
  build/final-review/generated/MINI/atax --wrapper thin
```

For a fresh compiler build and lower:

```bash
bash examples/esp-validation/emilia/container.sh cmake --build build/plugin -j2
bash examples/esp-validation/emilia/container.sh \
  python3 -c 'from lit.main import main; main()' -j2 -sv build/plugin/test
bash examples/esp-validation/emilia/container.sh bash scripts/tosa_to_lcs.sh \
  build/final-review/generated/MINI/atax/01_tosa.mlir build/review-atax
bash examples/esp-validation/emilia/container.sh env \
  BAMBU_DEVICE=xc7vx485t-2ffg1761 BAMBU_CLOCK_PERIOD=20 \
  bash scripts/ll_to_verilog_esp_version.sh build/review-atax/05_hw_acc.ll forward_kernel
```

`LCS_OVERLAP=false` selects the serialized tiled schedule. `LCS_TILE_SIZES` can
replay a recorded selection; the default requires no per-kernel overrides.
The legacy `generated-lcs/run-adapter.sh` expects its older vector layout. Use
`native-dma/capture.py` and `run-component.py` for this new outlined-host flow.

For review, start with the admission/token lifetime and reuse inference commits,
then inspect the LARGE 2MM/3MM reports, and finally the tiling policy. Agreement
on these defaults and the thin adapter is the gate before regenerating LARGE
bitstreams and rerunning experiment 1.
