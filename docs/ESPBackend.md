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
scripts/esp_gemm_demo.sh    # all three configurations below, end to end
```

## The offload

`sb_cli/recipes/esp/transform.mlir` applies `sodap-linalg-batch-matmul-to-esp`,
a pass in `examples/soda-plugins`. It replaces each `linalg.batch_matmul` with
seven calls that mirror ESP's own invocation sequence:

```
esp_alloc_shared(total_bytes) -> handle
esp_float2fixed_f32(A, handle, off_in)      # host floats -> Q16.16 in shared memory
esp_float2fixed_f32(B, handle, off_w)
esp_accel_cfg_regs(M, K, N, off_in, off_w, off_b, off_o)
esp_accel_start()
esp_accel_wait()
esp_fixed2float_f32(handle, off_o, C)       # and back
esp_free_shared(handle)
```

The shared buffer is one flat region, `[ I | W | B | O ]`, with the offsets in
elements. `alpha` and `beta` are not part of it: PolyBench's gemm reaches
`linalg` as a `batch_matmul` plus two `linalg.generic` scalings, and only the
matmul is offloaded.

Two properties of the pass are worth knowing before reading a result:

- **There is no bias.** `batch_matmul` has none, so the pass points `FFN_ADDRB`
  at the output region. The accelerator still reads `N` bias words from there,
  which is why `esp_alloc_shared` zeroes the buffer rather than merely
  allocating it.
- **The batch dimension is ignored.** Sizes come from `dim(A,1)`, `dim(A,2)` and
  `dim(B,2)`. That is correct for these kernels — TOSA always gives them batch 1
  — and `EspRuntime.cpp` bounds-checks every transfer against the buffer rather
  than trusting it.

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
```

```
Called: esp_alloc_shared
	total_bytes=7400
Called: esp_float2fixed_f32
	src: rank=3, shape=1x20x30, elements=600
	offset=0
...
Called: esp_accel_cfg_regs
	seq_len=20, indim=30, outdim=25
	off_in=0, off_w=600, off_b=1350, off_o=1350
...
TEST FAILED (456/500 elements exceed tolerance 0.001)
```

**The mismatch is the expected outcome**, and the reason `make` still succeeds:
the mock does not compute, so the output buffer keeps whatever the surrounding
software left in it. What this configuration checks is everything around that —
that the pass fired, that the dimensions and offsets are right, that the
descriptor ABI matches, and that the kernel links against the runtime. `make
check` is the target that treats `TEST FAILED` as an error, so use it on the
configuration above and not on this one.

### `--backend esp` — the RISC-V object and the application

```bash
pixi run sb-cli init --benchmark_name gemm --flow transformed \
    --backend esp --stage object --output_dir esp_gemm_mini_test
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

`esp-app/` is self-contained: copy it into `<esp>/soft/$SOC/baremetal/` and
`make`. `--stage binary` does that for you when `ESP_ROOT` or `DRIVERS` points
at a checkout, and otherwise says exactly what is missing rather than failing
somewhere in the toolchain.

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
