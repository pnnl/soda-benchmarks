# Generated floating-point GEMM on ESP

This example records the integration used to run Tommaso's generated `host.ll`
and Bambu RTL together. The host and generated accelerator are unchanged. The
runtime additions are in `examples/soda-plugins/lib/sodap/ExecutionEngine/`;
the SystemVerilog here is an ESP-side adapter, outside the generated core.

This is a validation example, not an automatic ESP SoC generator or a general
replacement for ESP's socket wrappers. The existing FFN/Q16.16 lowering and
its defaults remain available. The new host lowering that produced the archive
is not reconstructed by this example.

## Integration changes

* `esp_copy_to_shared_f32(rank, descriptor, handle, offset)` and
  `esp_copy_from_shared_f32(handle, offset, rank, descriptor)` copy raw IEEE f32
  representations. They support rank-zero scalars and strided memrefs, including
  nonzero descriptor offsets. Shared offsets are **32-bit element indices**.
  There is no float-to-fixed conversion in these functions. Invalid handles or
  copy ranges print an error and leave the destination unchanged; descriptors
  and their backing storage must still be valid.
* `SODAP_ESP_DEVICE_ID` and `SODAP_ESP_DEVICE_NAME` select the device when building
  `EspRuntime.cpp`. Defaults remain the existing FFN device. The example uses
  `0x07a` / `sld,sqr_bambu`, the identity of its prepared validation socket.
  This is a build-time choice, not a generated hardware metadata interface.
* The runtime places a RISC-V memory/I/O fence before START and after acknowledging
  DONE. Cache maintenance remains the existing ESP runtime's responsibility.
* `sqr_bambu_basic_dma64.sv` unpacks `(length_words << 32) | index_words`, converts
  32-bit word indices to 64-bit DMA beat indices, and selects the correct half
  of each read beat. It supports the one-word reads of `alpha` and `beta`,
  including a scalar in the upper half of a beat. Writes must cover aligned,
  whole 64-bit beats; arbitrary partial writes are not implemented.
* The wrapper registers `conf_done` for one cycle before starting the generated
  core. This lets the ESP socket complete configuration before the first DMA
  request. It also waits for its write adapter to drain before exposing DONE.
* Write descriptors are accepted locally and issued to ESP only after a complete
  output chunk is buffered. Larger writes are split into bounded chunks. This
  prevents an early write command from occupying ESP's native DMA engine while
  the core still needs reads to produce the output; see
  [the GESUMMV regression](../../../docs/ESPNativeDMAWriteBuffer.md).

The runtime still permits **one live shared buffer**. Sequential alloc/run/free
calls work, but ESP's baremetal bump allocator may not reclaim memory on free.
This update does not add concurrent handles, device scheduling or multi-offload
IR support.

## Workload and input artifacts

The operation is `out = alpha * (A @ B) + beta * C`, with `A[8][16]`,
`B[16][12]`, and `C/out[8][12]`. The core uses 4-by-4 tiles and ping-pong buffers.
Its shared-memory layout, in f32 elements, is:

| Region | Offset | Elements |
|---|---:|---:|
| C | 0 | 96 |
| A | 96 | 128 |
| B | 224 | 192 |
| out | 416 | 96 |
| beta | 512 | 1 |
| alpha | 513 | 1 |

The host calls `alloc -> copy inputs/scalars -> start -> wait -> copy output ->
free`. It does not write shape CSRs: these dimensions and region offsets are
fixed in the generated core. The generic `esp_accel_write_reg` API remains
available for a future lowering that emits configurable hardware.

`Inputs/host.ll` is a byte-for-byte copy of the supplied host artifact. The
scripts accept an extracted `gemm/` directory with the following original files;
they neither edit nor regenerate them:

| File | SHA-256 |
|---|---|
| `host.ll` | `f1b22de032750f9fddd9b079da5a1796a0f7e50774f9915e110b741cd8c4cb15` |
| `bambu/forward_kernel.v` | `2ed1e8b1074ef1008eec937c9bd94f8e832848a60fdb7cffe3c5cecd23a1e75e` |
| `bambu/panda_libtech.v` | `3f5094703d1cd9f01da608ba6a2548e83aa516b740491df9e5ad519c464c5539` |
| `bambu/array_429488.mem` | `2f5d66176fd2f0f1757f227bbddb0b44a55364165bfa012ecf098c0c25e85aa8` |

`bambu/array_429488.mem` is also required by the generated RTL. The original
archive and generated RTL are external inputs, not vendored here. The matrix
fixtures use finite binary fractions and an independent row-major GEMM oracle;
all intermediate values are exactly representable for these particular tests.
Bit-exact agreement here is not a general floating-point error guarantee.
Both replay scripts generate their fixtures in the run directory. To inspect
them separately, run `python3 examples/esp-validation/generated-gemm/make-vectors.py
/tmp/gemm-fixtures`. The numerical arrays are generated, not vendored.

## Hardware-free regression tests

The normal plugin lit suite contains `esp-copy-f32.mlir` and
`run-copy-f32.mlir`. The first checks scalar bit patterns, strided and reversed
views, empty shapes and bounds. The second lowers actual MLIR memref descriptors
and calls the copy API through `mlir-cpu-runner`. The mock performs real raw-f32
copies but still does not compute GEMM.

```sh
lit -v --filter='sodap/ESP/' /path/to/plugin-build/test
```

## Adapter simulation

With Icarus Verilog installed, from the repository root:

```sh
bash examples/esp-validation/generated-gemm/run-adapter.sh /path/to/gemm
```

This checks the supplied core and wrapper against a nontrivial generated fixture,
while withholding write-request acceptance until cycle 12000. Expected:
`GEMM_DMA64 PASS errors=0`, 96 exact outputs, 218 read requests and 24 write
requests. It is a component test; it does not execute a RISC-V CPU or the ESP NoC.

## Full-system replay

`run-system.sh` rebuilds the baremetal executable using **this branch's C++
runtime**, compiles the unchanged supplied core with the wrapper, and runs a
RISC-V application through the ESP socket, TLB, NoC and memory tile.

It requires a compatible, already-compiled ESP validation SoC: Ariane RV64,
64-bit DMA, scatter/gather enabled, one memory tile, L2/LLC disabled, and the
`sqr_bambu_basic_dma64` wrapper interface/device identity. It also requires
ModelSim with its vendor libraries and license configured, ESP baremetal driver
libraries, a startup that clears `.bss`, `clang-16`, and
`riscv64-unknown-elf-{gcc,g++,nm,objcopy}` on PATH. This repository does not supply
the ESP checkout, compiled SoC, vendor libraries or a new FPGA bitstream.

```sh
bash examples/esp-validation/generated-gemm/run-system.sh \
  /path/to/gemm /path/to/esp /path/to/compiled-esp-soc
```

The replay overrides this VC707 testbench's `mig_axiram/kbytes` generic to
16384 (16 MiB). The original 2 MiB RAM model aliases higher addresses: the
runtime's 1 MiB minimum allocations plus ESP's probe buffers otherwise wrap
onto application storage. The simulated RAM must cover the application's
physical address ranges without such aliasing. This is a testbench capacity
setting, not a generated-accelerator change.

Optional variables: `CLANG`, `RISCV_PREFIX`, `GEMM_STALL` (default 50000),
`ESP_SIM_RAM_KBYTES` (default 16384),
`ESP_MODELSIM_INI` and `ESP_MODELSIM_WORK`. The latter two allow using matching
external simulator libraries. Use a fresh output directory on each run.
Existing compiled SoC libraries are copied; the source SoC is not modified.

The application invokes the host three times: all-one inputs, then two different
signed fractional datasets and alpha/beta pairs. It compares 288 output words
and checks that input arrays/scalars are unchanged. The wrapper additionally
checks transfer counts and drains. Success requires three zero-error case lines
and `GEMM_SYSTEM PASS total_errors=0`, not just a zero simulator exit status.
The testbench's final `Program Completed!` assertion is its normal exit path.

The artificial stall is a correctness stress test, not a performance result.
This example does not validate coherent DMA, partial writes, arbitrary shapes,
multiple memory tiles, or FPGA timing/deployment.

## Recorded validation (3 November 2026)

The refreshed branch passed 42 plugin lit tests, 139 CLI unit tests, a fresh
GEMM MINI RISC-V object build, and the raw-f32 memory-safety checks. The component
simulation produced 96 bit-exact outputs. The full-system run used this branch's
C++ runtime and produced the following results:

| Case | Output errors (96 words) | Input errors | Core/adapter cycles, including stall |
|---|---:|---:|---:|
| 0 | 0 | 0 | 58120 |
| 1 | 0 | 0 | 58124 |
| 2 | 0 | 0 | 58124 |

All 288 output words matched exactly. The [selected validation log](validation.log)
records these checks and input/software identities. The 50000-cycle write-request
stall makes these correctness results, not a performance comparison. No new FPGA
run is claimed.
