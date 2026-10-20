# The `cpu` and `esp` backends for `sb-cli`

`--backend` picks what consumes the LLVM IR soda-opt produces. `bambu`
synthesizes it. The two backends described here compile it instead:

- **`cpu`** links it into a native executable and runs it, checking the result
  against PyTorch. Nothing outside this container is needed.
- **`esp`** cross-compiles it for the Ariane (CVA6) RISC-V core an
  [ESP](https://esp.cs.columbia.edu) SoC is built around, and stages a
  self-contained baremetal application beside it. Whichever
  `linalg.batch_matmul` the kernel contains is offloaded to an ESP accelerator.

```bash
scripts/esp_gemm_demo.sh    # the three container configurations below
```

See [ESP validation](ESPValidation.md) for runnable checks, the companion
`soda-opt` patch, and the additional dependencies used for the FPGA run.

## The offload

`sb_cli/recipes/esp/transform.mlir` applies `sodap-linalg-batch-matmul-to-esp`,
a pass in `examples/soda-plugins`. It replaces each `linalg.batch_matmul` with
calls that mirror ESP's own invocation sequence:

```
esp_alloc_shared(total_bytes) -> handle       # zeroed
esp_float2fixed_f32(A, handle, off_in, K)     # host floats -> Q16.16, rows K apart
esp_float2fixed_f32(B, handle, off_w, Npad)   # rows Npad apart
esp_accel_write_reg(reg, value)  x7           # one call per register
esp_accel_start()
esp_accel_wait()
esp_fixed2float_f32(handle, off_o, Npad, C)   # and back, dropping the padding
esp_free_shared(handle)
```

The pass takes three options, set in the transform schedule: `vec-len`, the
accelerator's vector length; `profile`, which brackets the pack, accelerator,
unpack and epilogue phases with `esp_prof_begin`/`esp_prof_end` (see
`esp_prof.h`); and `marshal`, below.

### `marshal=ir` — the conversion as IR

With `marshal=runtime` (the default) the two copies are runtime calls, and the
loop and the Q16.16 format are C++. With `marshal=ir` the pass generates them:
`esp_alloc_shared` returns the buffer as a `memref<i32>`, each operand becomes
a `memref.reinterpret_cast` view of it — the padded stride is the view's
stride, the padding columns are outside it — and the conversion is a two-op
`linalg.generic` over the view. The token format is one struct in the pass,
`FixedPointToken`; the runtime never sees a datatype.

```
%mem = call @esp_alloc_shared(%c8928_i64) : (i64) -> memref<i32>
%vW  = memref.reinterpret_cast %mem to offset: [600], sizes: [1, 30, 25],
         strides: [960, 32, 1]                        # rows Npad=32 apart
linalg.generic ins(%B) outs(%vW) { mulf 65536.0 ; fptosi }
```

Because the conversion is ordinary IR, later transformations see it. The
`esp-ir` recipe follows the pass with `fold-memref-alias-ops`,
`affine-loop-fusion` and `affine-scalrep`; on gemm that fuses the unpack, the
dead zero-fill of C and the three alpha/beta `linalg.generic`s into one loop
that reads tokens straight from the shared buffer. `esp_free_shared` moves to
the end of the block in this mode, because fusion may sink the unpack loop
into its consumer. The pass conservatively permits at most one IR offload per
function and requires static shapes. Use runtime marshalling for sequential
offloads. Reentrant calls that offload while a buffer is live are unsupported;
the guard is not an interprocedural lifetime analysis.

Measured on the VC707 (Ariane at 50 MHz, gemm MINI, cycles from `mcycle`;
identical results in all three):

| | `marshal=runtime` | `marshal=ir` | `esp-ir` recipe (fused) |
|---|---|---|---|
| pack | 60,436 | 57,239 | 57,253 |
| accel | 74,027 | 74,071 | 74,132 |
| unpack | 39,643 | 38,736 | 66 |
| epilogue | – | 79,659 | 61,483 |

Generating the conversion is performance-neutral by itself; fusion halves
unpack+epilogue. Pack — 42 cycles per element into the uncached DMA buffer — is
still a substantial host cost, and has no CPU consumer to fuse with; removing it
means allocating the operands in shared memory in the first place, which the
views make possible. Under the bare-pointer convention a rank-0 memref is a
single pointer, so both modes link against the same runtime.

**The pass owns the layout and accelerator register programming.** The runtime
still selects the FFN device and supplies Q16.16 helpers for runtime marshalling;
it is not yet a fully accelerator-independent implementation.
The shared buffer is one flat region, `[ I | W | B | O ]`, with offsets in
elements and `Npad = roundup(N, vec-len)`:

- **N is padded to `Npad`.** The accelerator computes whole `vec-len`-wide
  output tiles (`w_iter = outdim / vec-len`, an integer division in hardware),
  so W and O are laid out with the padded stride and the register gets `Npad`.
  The `ld` argument to the copies carries that stride; the runtime copies row by
  row and needs no notion of why.
- **The bias has its own region, and stays zero.** `batch_matmul` has no bias,
  but the accelerator re-reads its bias once per output tile while the output
  is being written, so the two cannot alias. `esp_alloc_shared` zeroes the
  buffer, which is what a bias-less matmul needs.
- **The register map lives in the pass**, as constants that describe
  `sld,ffn_sysc_catapult`. The runtime writes whatever `(offset, value)` pairs
  it is handed. When the accelerator is generated, those constants become an
  output of that generation.
- **Batch must be statically known to be 1 for every operand.** Larger or
  dynamic batches are rejected before rewriting, because the layout has no
  batch dimension. Operands must be rank-3 f32 memrefs.

`alpha` and `beta` are not part of any of this: PolyBench's gemm reaches
`linalg` as a `batch_matmul` plus two `linalg.generic` scalings, and only the
matmul is offloaded.

## Two runtimes

The seven symbols are declared once, in
`examples/soda-plugins/include/sodap/ExecutionEngine/ESPRuntime.h`, and
implemented twice:

| | `EspRuntimeMock.cpp` | `EspRuntime.cpp` |
|---|---|---|
| Built | always, as `libmlir_mockesp_runner_utils.so` | only with `-DSODAP_ENABLE_ESP_RUNTIME=ON -DSODAP_ESP_ROOT=<esp>`, or from the staged `esp-app/` |
| Does | prints each call, its arguments and the decoded operand shapes | probes `sld,ffn_sysc_catapult` (0x074), allocates the shared buffer and its page table, converts to Q16.16, programs the registers, starts and polls the accelerator |
| Computes | **no** | yes |

Both decode the `memref<*xf32>` argument the same way, through
`sodap::decodeMemRefF32` in the header — which is what makes a host run a real
check of the descriptor ABI rather than a smoke test.

## Three configurations

### `--backend cpu` with no ESP schedule — the numeric check

```bash
pixi run sb-cli init --benchmark_name PolyBenchPyTorch.linear_algebra.blas.gemm \
    --dataset MINI --dtype float32 --flow baseline \
    --backend cpu --stage simulation --output_dir cpu_gemm_mini_ref
cd benches/experiments/cpu_gemm_mini_ref && pixi run make && pixi run make check
```

```
kernel=gemm dataset=MINI dtype=float32 elements=500 tol=0.001
TEST PASSED (max error: 1.90735e-06)
```

This validates the whole path — torch, TOSA, linalg, soda-opt, LLVM, native
code — against PyTorch's own answer for the same inputs. It works for any
`--flow`.

### `--backend cpu --instrumentation esp` — the offload, on the host

```bash
pixi run sb-cli init --benchmark_name gemm --flow transformed \
    --instrumentation esp --backend cpu --stage simulation \
    --output_dir cpu_gemm_mini_esp
pixi run make -C benches/experiments/cpu_gemm_mini_esp
```

```
Called: esp_alloc_shared
	total_bytes=8928
Called: esp_float2fixed_f32
	src: rank=3, shape=1x20x30, elements=600
	offset=0, ld=30
Called: esp_float2fixed_f32
	src: rank=3, shape=1x30x25, elements=750
	offset=600, ld=32
Called: esp_accel_write_reg
	offset=0x58, value=20
...
Called: esp_accel_write_reg
	offset=0x40, value=1592
...
TEST FAILED (456/500 elements exceed tolerance 0.001)
max error = 3600064035 e-9, tolerance = 1000000 e-9
---------------------------------
region        cycles       calls        mean     share
------------------------------------------------------
total              20438       1       20438     100%
pack                2495       1        2495      12%
accel                711       1         711       3%
unpack               861       1         861       4%
```

**The mismatch is the expected outcome**, and the reason `make` still succeeds:
the mock does not compute, so the output buffer keeps whatever the surrounding
software left in it. What this configuration checks is everything around that —
that the pass fired and that the kernel links against the runtime. Inspect the
printed shapes and offsets; the automated assertions live in the focused
MLIR/mock-runner test. This is not an assertion-based end-to-end numerical test. `make
check` is the target that treats `TEST FAILED` as an error, so use it on the
configuration above and not on this one.

### `--backend esp` — the RISC-V object and the application

```bash
pixi run sb-cli init --benchmark_name gemm --flow transformed \
    --backend esp --stage object --output_dir esp_gemm_mini_test
pixi run make -C benches/experiments/esp_gemm_mini_test
```

`--backend esp` selects the ESP schedule on its own; `--instrumentation` still
overrides it. The result:

```
output/esp/transformed/06_kernel_riscv.o     ELF 64-bit LSB relocatable, UCB RISC-V
output/esp/transformed/esp-app/
    <experiment>.c    the driver
    testdata.h        inputs and the PyTorch golden
    kernel.o          the same object
    EspRuntime.cpp    } the runtime the esp_* calls resolve to
    ESPRuntime.h      }
    Makefile          APPNAME + include $(DRIVERS)/common_bare.mk
    README.md
```

`esp-app/` stages the application sources and object, but still depends on an
ESP checkout, matching SoC support files, startup/linker configuration and a
RISC-V toolchain. For the validated ESP revision, copy it under
`<esp>/soft/common/apps/baremetal/<appname>` and build from the SoC directory as
described in [ESP validation](ESPValidation.md). The legacy `--stage binary`
helper assumes a different ESP directory layout and was not used for this
validation; use the documented manual link for this checkout.

## What each stage builds

| `--stage` | `bambu` | `cpu` | `esp` |
|---|---|---|---|
| `llvm` | `output/05_llvm_<flow>.ll` | same | same |
| `object` | — | — | `output/esp/<flow>/06_kernel_riscv.o` + `esp-app/` |
| `binary` | — | `output/cpu/<flow>/06_kernel` | `output/esp/<flow>/07_kernel.riscv` |
| `verilog` | `output/bambu/<flow>/06_verilog.v` | — | — |
| `simulation` | `output/bambu/<flow>/07_results.txt` | `output/cpu/<flow>/07_results.txt` | — |
| `gds` | ORFS `6_final.gds` | — | — |

A pair with no entry is rejected by `sb_cli.flow.resolve_target`, naming the
stages that backend does reach.

## The test data

The driver is one fixed file, `scripts/lib/soda_testbench_main.c`. Everything
kernel-specific is in the `output/testdata.h` that `torchscript.py
--emit-testdata` generates: the inputs from the kernel's own `init_array`, the
golden from running the PyTorch model on them, and the macros `TD_CALL` and
`TD_DECL` that spell the call, since C cannot express a variable arity.

`TD_CALL` is where a subtlety lives. `-soda-outline-bambu-code` orders
`forward_kernel`'s parameters **by first use inside the outlined region**, not by
`@forward`'s signature — for gemm it turns `(alpha, beta, C, A, B, out)` into
`(A, B, alpha, beta, C, out)`. Calling in signature order compiles, runs, and
quietly computes something else. `scripts/kernel_arg_order.sh` reads the
permutation off the `soda.launch_func` the outliner leaves behind, and the
Makefile rule hands it to `--emit-testdata`.

This needs a benchmark, so an experiment scaffolded without `--benchmark_name`
cannot use either backend — `torchscript_default.py` has no `init_array` to draw
inputs from.

`--dtype float16` is rejected: it has no portable C spelling and the kernel is
compiled as-is rather than through a half-float library. `float32` and `float64`
both work.

## soda-opt

The transformed flow gained one pass for this,
`--reconcile-unrealized-casts` in `scripts/soda_to_llvm_transformed.sh`. Calls
taking `memref<*xf32>` lower correctly under the bare-pointer convention, but
the convention leaves the operand's cast behind as a dead
`builtin.unrealized_conversion_cast`, which `mlir-translate` rejects. It is a
no-op for a schedule that introduces no such calls.

## Toolchain

| | Provided by |
|---|---|
| host compile and link (`cpu`) | `clang` on `PATH`; override with `CC` |
| RISC-V object (`esp`) | the same `clang` — `--target=riscv64-unknown-elf -march=rv64imafdc -mabi=lp64d`, overridable as `RISCV_TARGET` / `RISCV_MARCH` / `RISCV_MABI` |
| the ESP runtime | an ESP checkout: `esp_accelerator.h`, `esp_probe.h`, `probe()`, `aligned_malloc()`, `common_bare.mk` |
| the mock runtime | `examples/soda-plugins/build/lib/`; override with `ESP_RUNTIME_LIB` |

Only the third is missing from this container, which is why `object` is the
deepest `esp` stage that can be built here.

## Not covered

`sb-cli collect`'s collectors all read `output/bambu/**`, so they find nothing
for a cpu or esp run. Collecting the max error, the pass/fail and the launch
count is a natural next step.
