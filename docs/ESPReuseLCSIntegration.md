# Runtime reuse with Tommaso's task-based LCS flow

This integration combines Tommaso's `feature/esp-lcs` at
`33266a989673d7584887fa0c7037b052883837ed` with the validated local runtime
optimizations at `639e8fbe425f8bc945b02b912aabbc5720845063` and reusable GEMM/NN
support. Merge `f4bc51f` integrates the committed histories; `e9965ab` ports the
previously local reusable-operator source. The original measured checkout and
published `handoff/esp-reuse-resnet18-20261106` branch are preserved separately.

## What is shared, and what remains operator-specific

Tommaso's default flow creates compute tasks on tensors, tiles and bufferizes
them, creates load/store tasks, schedules their graph, and inserts ping-pong
storage before LLVM lowering. That path remains the default for fixed graphs.

The reusable GEMM and convolution generators now have optional `lcs` schedules.
They construct three Dataflow tasks with matching runtime tile traversals:

1. Load reads the descriptor once, broadcasts it to compute/store, and fills
   local input tiles through the single DMA reader.
2. Compute consumes input tiles and accumulates privately, preserving the
   sequential implementation's floating-point reduction order.
3. Store consumes completed output tiles through the single DMA writer.

These generators reuse the common **PingPong, TasksToNodes and NodesToFunc**
passes. They do not run the generic fixed-graph Schedule pass on dynamic loops.
Matching dynamic producer/consumer traversals are established by generator
construction; the shared pass does not prove arbitrary dynamic schedules safe.

Shared tiles become two slots with depth-one ownership token streams. The new
`sodap.dataflow_tile` attribute marks an `scf.execute_region` containing all
scalar accesses for one tile transaction. Ping-pong acquires/releases ownership
around that region; it must run before canonicalization inlines the region.
Mixed marked/unmarked accesses to the same shared buffer are rejected. Existing
unmarked fixed-graph task behavior is unchanged.

GEMM has shared A, B, old-C and result tiles. When beta is zero, load supplies a
zero old-C tile without a C DMA read, keeping the token sequence uniform. Every
stage drains the descriptor even when its magic/version is invalid.
Convolution retains separate weight banks and private accumulators for each
output lane, plus shared activation, bias and result tiles. Addition remains
sequential. All host offloads still complete sequentially.

## Runtime parameters stay independent of scheduling

Invocation descriptors, runtime validation, tensor locations and host calls are
unchanged. Dimensions, strides, coefficients and addresses are runtime values;
tiles, lane counts, storage and internal scheduling are hardware choices.

Select `LCS_GEMM_SCHEDULE=lcs` for standalone GEMM, or
`NN_GEMM_SCHEDULE=lcs NN_CONV_SCHEDULE=lcs` for the neural-operator generator.
Both default to `sequential`, preserving the existing baseline. The generated
`schedule.txt` records these choices. Hardware generation takes no model input.

The GEMM host LLVM, RV64 object, CPU reference LLVM, ABI header and layout are
byte-identical between sequential and LCS implementations with 4×4×4 tiles.
The same host LLVM/ABI also works with an LCS 2×8×3 implementation. Digests are
retained in [host-identity.json](https://github.com/WilliamBaisi/bambu-esp-example/tree/1cbbf1379a004079d43ef3064a34226bddecebea/integration/soda/docs/validation/esp-reuse-lcs-20261107/host-identity.json).

## Validation on the integration

Evidence is under [validation/esp-reuse-lcs-20261107](https://github.com/WilliamBaisi/bambu-esp-example/tree/1cbbf1379a004079d43ef3064a34226bddecebea/integration/soda/docs/validation/esp-reuse-lcs-20261107/).
Its `SHA256SUMS` covers saved logs and reports; historical absolute paths in
those files identify the actual run and are not portable configuration.

| Check | Result |
| --- | --- |
| Plugin lit suite, including runtime tile ownership and mixed-scope rejection | 67/67 pass |
| Neural graph lowering unit tests | 7/7 pass |
| Default SMALL flow: GEMM, SYR2K, TRMM, SYMM, SYRK, GESUMMV, DOITGEN, THREEMM, ATAX, TWOMM | Host/accelerator LLVM, RV64 object and native CPU numerics pass |
| Default unsupported cases | GEMVER fan-out and BICG/MVT multiple outputs still explicitly rejected |
| Sequential and LCS GEMM component RTL | Two dependent calls and ten-call coefficients/row-strides/accumulation suite pass, with normal DMA and 50,000-cycle initial stalls |
| LCS GEMM 2×8×3 tiles | Same two-call and ten-call fixtures pass under both stall settings |
| LCS GEMM invalid descriptors | Bad magic, bad version, then two valid calls pass without reset or unintended writes |
| Sequential and LCS GEMM full-system RTL | Two launches on instance 0, shared intermediate address, 98 correct outputs |
| Conv/add/GEMM native-host/RTL-accelerator cosimulation | All pass with DMA stall masks 0, 1 and 7; accelerator results checked against oracle |
| Wide convolution 3→19→23→5 channels | Multiple input-channel tiles plus tails pass with stall masks 0 and 7 |
| Conv/add/GEMM full-system RTL | 3/2/2 launches; 60/143/6 outputs; zero numerical error |
| Complete ResNet-18 native execution | 30 offloads, 1000 logits, maximum absolute error 5.781650543212891e-6 |
| Complete ResNet-18 FPGA execution, newly built software | 30 offloads, 1000 logits, zero mismatches, maximum absolute error reported as 5782e-9 |

The ten default kernels' native checks validate the CPU reference, not the new
accelerator schedule. Their prior upstream GEMM/TWOMM RTL checks are preserved
in the separate review. This integration's fresh hardware checks concern the
reusable engines listed above.

For the small two-GEMM application, identical CPU software gives:

| Measurement | Sequential | LCS | Sequential / LCS |
| --- | ---: | ---: | ---: |
| Component accelerator cycles, two calls, no initial stall | 7,041 | 5,316 | 1.324× |
| Full-system accelerator boundary cycles, two calls | 12,991 | 10,425 | 1.246× |
| Full-system complete kernel CPU cycles | 564,504 | 561,847 | 1.0047× |

The total kernel improves only 0.47% because setup and CPU copies dominate this
small application. Component and system boundary cycle counts have different
memory/transport conditions and must not be substituted for whole-model time.

## Complete network on VC707

The integrated three-engine FPGA design meets timing at the existing 50MHz
CPU/accelerator clock: setup slack is +0.098357ns and hold slack +0.053293ns.
The bitstream SHA256 is
`6762fda9b5391d51bff04b536df89af0236109fb3ce91df39996095e37591f03`.
It passed all 30 offloads and 1000 output comparisons using the newly compiled
software. Input, pretrained parameters, host LLVM and invocation plan are
byte-identical to the original run. The complete executable differs in its
runtime object, so a second run with the preserved baseline executable is used
for the controlled hardware comparison.

| Complete inference | CPU cycles | Time at 50MHz |
| --- | ---: | ---: |
| Historical sequential engines | 7,669,179,138 | 153.3836s |
| LCS engines, identical preserved executable | 5,803,106,935 | 116.0621s |
| LCS engines, freshly compiled software | 5,804,799,476 | 116.0960s |

The controlled run also passes all 30 offloads and 1000 outputs. It is **1.322×
faster** and reduces timed inference by **24.33%** with an identical executable
(SHA256 `2bffa2130a38cad9f59bbeaf2a01757e2ff9a0aef8691be83edda99d92df87d5`).
The [comparison report](https://github.com/WilliamBaisi/bambu-esp-example/tree/1cbbf1379a004079d43ef3064a34226bddecebea/integration/soda/docs/validation/esp-reuse-lcs-20261107/resnet18/controlled-comparison.json)
records the exact measurements. The fresh integrated run is 1.321× faster and
reduces timed inference by 24.31%.
The timed region includes host work and accelerator execution; programming,
image loading and output validation are outside it, as in the baseline. The
historical scalar CPU-only measurement is 90,055,418,413 cycles (1801.1084s),
15.51× the fresh integrated run. That CPU result was not rerun in this campaign
and uses the earlier documented scalar reference, not an optimized CPU library.
The deterministic input validates numerical equivalence, not ImageNet accuracy.

Pipelining has a resource cost. These are complete-SoC totals from routed
implementations, including CPU, interconnect and memory infrastructure:

| Resource | Sequential design | LCS design |
| --- | ---: | ---: |
| LUTs | 137,310 | 142,356 |
| Flip-flops | 132,572 | 136,936 |
| BRAM36 tiles | 92.5 | 111.5 |
| DSPs | 220 | 242 |

This is one schedule comparison with unchanged local tile/lane settings, not a
search for an optimal FPGA architecture. Timing, utilization and board evidence
are preserved with the report.

## Review checkout and preserved evidence

The source changes are presented as focused commits based on Tommaso's
`33266a9`. The original integration at `85ec87d` and its measured build outputs
remain preserved separately. The [emilia tutorial](ESPSynthesisSimulation.md)
identifies the new independent review checkout and provides source-contained
helpers for building and simulating it. The review also fixes Make dependency
tracking for `LCS_GEMM_SCHEDULE`, so changing schedules regenerates the hardware.

Raw results are stored in the linked, immutable publication snapshot rather
than copied into this compiler PR. Access to that companion repository may be
required; the tables and artifact hashes in this document summarize the
validation without requiring those logs.

## Reproduction

The detailed commands remain in the [GEMM guide](../examples/esp-validation/reusable-gemm/README.md)
and [neural-operator guide](../examples/esp-validation/reusable-nn/README.md).
Use fresh output directories and the plugin built from this source tree.

```bash
# Inside the configured compiler environment:
NN_CONV_SCHEDULE=lcs NN_GEMM_SCHEDULE=lcs \
  bash scripts/reusable_nn/generate_hardware.sh build/replay/hardware
python3 scripts/reusable_nn/combine_rtl.py build/replay/hardware
python3 examples/esp-validation/reusable-nn/export_operators.py build/replay/operators
# Follow the neural-operator guide to compile/stage the three apps, run their
# full-system checks, and only then package/implement the matching RTL.
```

The compiler environment used SODA image `agostini01/soda-trixie:v19.03`, LLVM19
and PandA2026.06. System tests used the preserved Ariane VC707 fixture with
DMA64, disabled L2/LLC, 16MiB simulation RAM, ModelSim and RV64 bare-metal
software. Full-network FPGA software uses the 1GiB board memory map.

## Remaining optimization and generality work

This integration does not automatically infer reusable contracts for arbitrary
operators, choose how many physical engines a model needs, or tune tiles and
banking. It provides selectable implementations behind the existing contracts.
The known default scheduler limits on fan-out/multiple outputs remain. Its
TWOMM intermediate recomputation/storage tradeoff and the unconnected
`sodap-linalg-constants-to-indices` pass remain separate follow-up work.
ResNet residual reuse continues through sequential host calls and shared arena
lifetimes; it does not demonstrate fan-out support in the fixed-graph scheduler.
