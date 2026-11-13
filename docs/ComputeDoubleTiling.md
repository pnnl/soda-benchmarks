# Compute: a second tiling level

Status of 2026-11-13. This covers making the compute tasks faster with a second, register-level tiling, measured on gemm SMALL. Files are in [compute_tiling/](compute_tiling/).

The flow already tiles once, for memory: `transform.sodap.tile_and_fuse` and its DMA cost model, see docs/LinalgToLCSStatus.md, "Tile sizes". The second level cuts each memory tile into small register blocks. It is done by hand here and isn't part of the flow yet.

## The problem

gemm SMALL is C(60×70) = α·A(60×80)·B(80×70) + β·C. The cost model picks the memory tile `[[20, 0, 20], [20, 0]]`, so one compute step of the matmul task (`forward_kernel_node2`) is:

```
for i in 0..20, j in 0..70, k in 0..20:   C[i][j] += A[i][k] * B[k][j]
```

- **Work per step:** 28,000 multiply-adds (MACs).
- **Load time per step:** 4360 cycles to load the next step's tiles in the meantime: A 20×20 in 1680 cycles, B 20×70 in 2680. This is the cost model's figure with its default DMA constants, which are not yet measured on the board.
- **Compute time today:** about 31–38k cycles per step, 7 to 9 times the load. Compute is the bottleneck.

## What Bambu does today

Read from `bambu.log` and Bambu's IR in `panda-temp/after_CSROA.ll`. The clock is 20 ns on the xc7vx485t.

- **The accumulator is already in a register.** Bambu's own LLVM passes keep C[i][j] in a register across the k loop (a `%.promoted` load, then `phi float`). The k loop is pipelined: II = 1, with an iteration latency of 3.
- **The FP adder takes one cycle** (1 state, no flip-flops, about 15 ns). The reduction `c += a*b` therefore adds a 1-cycle dependency from one k iteration to the next. That is never the limit below.
- **One FP multiplier and one FP adder** by default (`Constraining function ... with 1 resources`). Today that caps compute at 1 MAC per cycle, plus a few cycles of loop overhead for each (i, j).

So the plan is: more MACs per cycle, which needs more FP units, and enough memory bandwidth to feed them.

## The second tiling

Cut C into MR×NR blocks that stay in registers, with k as the block's outer loop and the block fully unrolled:

```
for ib in 0..20 step MR:
  for jb in 0..70 step NR:
    c[MR][NR] = C[ib.., jb..]          // MR·NR loads from the C buffer
    for k in 0..20:                    // pipelined by Bambu
      c[ii][jj] += A[ib+ii][k] * B[k][jb+jj]   // unrolled: MR·NR MACs
    C[ib.., jb..] = c                  // MR·NR stores
```

Each k iteration reads MR values of A and NR values of B, and does MR·NR MACs. Its II, in cycles, is the largest of these limits:

| Limit | Value |
|---|---|
| FP units, P of each kind | ⌈MR·NR / P⌉ |
| A's ports: a BRAM has 2, multiplied by A's number of banks | ⌈MR / (2·banks of A)⌉ |
| B's ports | ⌈NR / (2·banks of B)⌉ |
| The reduction | 1 (one adder latency) |

The block sizes must divide the memory tile: MR ∈ {2, 4, 5, 10, 20} and NR ∈ {2, 5, 7, 10, 14, 35, 70}.

**How many FP units:** give Bambu `P = ⌈MR·NR / II allowed by the ports⌉`. Fewer units than that make the units the limit. More units only add area. In every experiment below the unit count was derived this way from the block and the banking, except #4 and #5.

**Reading and writing C** costs MR·NR loads and MR·NR stores per block, for 1400 + 1400 accesses per step. On a 2-port buffer that is about 1400 cycles per step, whatever the block shape, because C accumulates across the 4 k-tiles of the memory level. Two ways to cut it:
- bank C;
- keep k whole at the memory level, so C starts at zero in registers and is written once (more on-chip memory for A and B).

## Experiments

### Set 1: the flow's gemm, Bambu only, not simulated

`compute_tiling/micro_tiling.sh` adds the second tiling to the flow's lowering, and the result goes through Bambu. The cycle numbers are **estimates** from the schedule (II, latency, block counts), not measurements. Area is Bambu's estimate for `forward_kernel_node2` only.

| # | Register block | FP units | Banking | II of the k loop (from Bambu) | What sets the II | Compute cycles per step (estimate) | Area of node2 |
|---|---|---|---|---|---|---|---|
| 1 | none (today) | 1 | none | 1 | 1 adder | ~31–38k | 6.4k |
| 2 | 5×7 | 1 | none | 35 | 1 adder | ~28k+ | 22k |
| 3 | 5×7 | 9 | none | 4 | B's ports (7 reads) | ~4.9k | 57k |
| 4 | 5×7 | 12 | none | 4 | B's ports | ~4.9k | 66k |
| 5 | 5×7 | 35 | none | 4 | B's ports | ~4.9k | 74k |
| 6 | 10×7 | 14 | none | 5 | A's ports (10 reads) | ~3.5k | 96k |
| 7 | 10×14 | 20 | none | not pipelined | — | — | 125k |

Notes on the rows:
- **#4 and #5** only check that more units can't beat the port limit. #4 adds nothing.
- **#7**: Bambu reports minII = 140 and "Loop pipelining not possible". The cause wasn't investigated.
- **Clock:** Bambu's estimated maximum clock stays at 53–63 MHz, above the 50 MHz target.

### Set 2: the same step in C, simulated with Verilator

Banking can't be done in the flow yet (see the next section). So the step is written in C, in `compute_tiling/mm.c`, with `#pragma HLS array_partition` and local 2-D buffers like the on-chip buffers. Every run matches the testbench's reference (0 mismatches).

- **Cycles** are measured as cycles(REP=2) − cycles(REP=1), which removes the copy-in and copy-out loops.
- **Area** is Bambu's estimate for the whole C function, copy loops included, so it isn't comparable to Set 1's area.

| # | Register block | FP units | Banking (cyclic) | II of the k loop | Compute cycles per step (measured) | Area of mm |
|---|---|---|---|---|---|---|
| 8 | 10×7 | 14 | none | 5 | **3504** | 99k |
| 9 | 10×7 | 14 | C: 7 banks (columns) | 5 | **2304** | 105k |
| 10 | 10×7 | 35 | A: 5 (rows), B: 7, C: 7 | 2 | **1064** | 186k |
| 11 | 10×7 | 70 | A: 10 (rows), B: 7, C: 7 | 1 | **664** | 316k |

How these confirm the model:
- **#8 matches #6.** The C version is representative of the flow: same II, and 3504 against the ~3.5k estimate.
- **#8 → #9:** banking C saves 1200 cycles, which is most of the ~1400 cycles of C reads and writes.
- **#10 and #11:** with A, B and C banked, the II drops to the FP-unit limit, 70 MACs / P.

## Results

1. **The register block is the main gain:** from about 31–38k cycles per step to 3504 (10×7, 14 units, no banking). That takes about 15× the compute task's area.
2. **The memory ports set the II, not the reduction.** Choose the block, then the banking, then P.
3. **With these DMA constants, 3504 is already under the 4360-cycle load,** so the load becomes the bottleneck. Banking (#9 to #11) makes compute faster still, but the kernel only gains if loading also gets faster. Before spending area on banking, measure the DMA on the board (`dma_request_cycles=64` is a guess).
4. **Area:** all area figures are Bambu's estimates, not Vivado. #11's 316k is above the VC707's ~300k LUTs. #9 (C banked) costs little.

## Banking in the flow: what is missing

Bambu partitions buffers through a marker call (CSROA, `etc/clang_plugin/ArrPart.cpp` in Bambu):

```llvm
call void @__bambu_csroa_partition__(ptr %buf, i32 2, i32 7, i32 2)
;                                    buffer   format: 0 complete, 1 block, 2 cyclic
;                                             factor (banks), dim (from 1; 0 means all dims)
```

`#pragma HLS array_partition` in C generates it. The buffer can be:
- an alloca or global of array type;
- a pointer argument, whose shape comes from `array_dims` in architecture.xml.

The partition spreads to every buffer passed through calls to the same memory, for example the top's ping-pong buffers.

**Why it fails on the flow's `.ll`:** CSROA needs each access to a partitioned buffer to be a single GEP from the base, with one index per dimension:

```llvm
%p = getelementptr [20 x [70 x float]], ptr %c, i32 0, i32 %i, i32 %j     ; accepted
```

The flow's memref→LLVM lowering emits a flat `getelementptr float, ptr %c, i32 %idx`, and Bambu's own passes (InstCombine, LICM) then split it into a base GEP plus `getelementptr i8, ptr %p, i32 4`-style offsets. Adding the marker to the flow's C accumulator fails with:

```
[CSROA] GEP does not address the object from its base with enough indices to select a partition
[CSROA]   offending user:   %23 = getelementptr float, ptr %14, i32 %22
```

C written with the row hoisted (`&C[i]`, then `[j]`) fails the same way. Clang's GEPs from `a[i][j]` on local arrays pass.

**What the flow needs, for the buffers to bank:**
1. Array-typed buffers: `alloca [20 x [70 x float]]`, not `alloca float, i64 1400`. That includes the top's ping-pong buffers that the partition spreads to.
2. One multi-index GEP per load and store, built from the `memref.load` and `memref.store` indices before they are flattened. A custom lowering pattern for those ops, for example in `sodap-convert-dataflow-to-llvm`.
3. The marker call for each banked buffer, with a factor and a dim per buffer: for the 10×7 block, A cyclic 10 on dim 1 (rows), B and C cyclic 7 on dim 2 (columns). For node arguments, the dim counts the ping-pong dimension: architecture.xml has `array_dims="2,20,70"`.
4. The load tasks' DMA write loops then write into the banks. CSROA picks the bank at run time where ScalarEvolution can't prove a single bank. In C that worked, at some area cost.

Start with C's local accumulator, item 1 on one alloca, which is #9's gain alone. A and B come next, through the top's buffers.

## Next steps

1. **Measure the DMA on the board.** If loading really takes ~4360 cycles per step, #8 (no banking) is enough for gemm.
2. **Make the second tiling a step of the flow:** apply `micro.mlir` from `scripts/lcs/transform.mlir` and add Bambu's `-C` option to the ESP Bambu options. Then simulate a whole kernel to confirm the Set 1 estimates.
3. **Banking in the flow**, as in the previous section.
4. **Choose the block automatically** with the formula above, giving `tile_and_fuse`'s cost model a compute term. No new user input: the II limits come from the block and the ports.
5. **Open points:**
   - why 10×14 isn't pipelined (#7);
   - other kernels: twomm and threemm have several matmuls, so tile them one by one;
   - the Columbia article on double tiling (not read).

## How to reproduce

### Set 1: the flow (devcontainer, then the host's Bambu)

```sh
# In the devcontainer: build gemm with the cost model's tile. Set 1 used dma_word_stride=1.
LCS_TILE_SIZES='[[20, 0, 20], [20, 0]]' LCS_DMA_WORD_STRIDE=1 \
  ./scripts/tosa_to_lcs.sh benches/experiments/gemm-SMALL-esp-object/output/01_tosa.mlir /tmp/gemm
# Without a block: must be identical to /tmp/gemm/tiles/forward_kernel/05_hw_acc.ll.
docs/compute_tiling/micro_tiling.sh /tmp/gemm /tmp/gemm_plain
docs/compute_tiling/micro_tiling.sh /tmp/gemm /tmp/gemm_10x7 10 7

# On the host (Bambu in /opt/panda), in the output directory:
source /opt/panda/settings.sh
bambu accelerator.ll --architecture-xml=architecture.xml --top-fname=forward_kernel \
  --compiler=I386_CLANG19 --generate-interface=INFER --generate-components=separate \
  --clock-period=20.0 --device=xc7vx485t-2ffg1761 -v3 --no-clean \
  -C='__float_mule8m23b_127nih=14,__float_adde8m23b_127nih=14' > bambu.log 2>&1
grep -E 'Initiation Interval|Iteration latency' bambu.log | grep node2
grep -A12 'Module binding information for function forward_kernel_node2:' bambu.log \
  | grep 'Total estimated area'
```

`micro_tiling.sh` repeats the "Tile: to nodes and LLVM" step of `scripts/tosa_to_lcs.sh`, so keep the two in sync. The identity check above catches drift.

Checked on 2026-11-13 at commit 678b967:
- without a block, `accelerator.ll` and `architecture.xml` are byte-identical to the flow's;
- with 10 7, Bambu gives row #6 again (II = 5, latency 9, area 96014).

### Set 2: the C kernel (host only)

```sh
# <outdir> <FP units> [-D options of mm.c]; prints the IIs, cycles, mismatches and area.
docs/compute_tiling/run_c.sh /tmp/c_none_r1 14 -DREP=1
docs/compute_tiling/run_c.sh /tmp/c_none_r2 14 -DREP=2     # compute = r2 − r1 cycles
docs/compute_tiling/run_c.sh /tmp/c_c7_r1   14 -DPC=7 -DREP=1
docs/compute_tiling/run_c.sh /tmp/c_all_r1  70 -DPA=10 -DPB=7 -DPC=7 -DREP=1
```

Don't partition `mm`'s arguments directly: that changes the interface the testbench drives, and the simulation then mismatches. That's why the buffers are local copies.
