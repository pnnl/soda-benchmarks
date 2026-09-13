# ESP accelerator offload, conversion in IR

The `esp` recipe with `marshal=ir`, plus an affine schedule that fuses what that
makes visible. Read `../esp/README.md` first; only the differences are here.

## What changes

With `marshal=ir` the pass does not call `esp_float2fixed_f32` /
`esp_fixed2float_f32`. It has `esp_alloc_shared` return the buffer as a
`memref<i32>`, carves each operand out of it with `memref.reinterpret_cast` —
the padded stride is the view's stride, the padding columns are outside it —
and emits the conversion as a two-op `linalg.generic`:

```
%mem = call @esp_alloc_shared(%bytes) : (i64) -> memref<i32>
%vW  = memref.reinterpret_cast %mem to offset: [600], sizes: [1, 30, 25],
         strides: [960, 32, 1]                    # rows Npad=32 apart
linalg.generic ins(%B) outs(%vW) { mulf 65536.0 ; fptosi }
...
%vO  = memref.reinterpret_cast %mem to offset: [1592], ...
linalg.generic ins(%vO) outs(%C) { sitofp ; mulf 2^-16 }
```

The token format is one struct in the pass, `FixedPointToken`; the runtime
never sees a datatype. `esp_free_shared` moves to the end of the block, because
fusion may sink the unpack loop into its consumer and the buffer has to outlive
wherever it lands. The current pass therefore conservatively allows at most
one offload per function in this mode. It requires static shapes and batch=1;
larger or dynamic batches are rejected in both marshalling modes. Use
`marshal=runtime` for sequential offloads. Interprocedural/reentrant offloads
while a buffer remains live are unsupported.

The schedule then runs `convert-linalg-to-affine-loops`,
`fold-memref-alias-ops` (so fusion sees one memref through the reshapes the
torch lowering wraps around every matmul), `affine-loop-fusion`, and
`affine-scalrep`. On gemm the unpack, the dead zero-fill of C, and the three
alpha/beta `linalg.generic`s become a single loop reading tokens straight from
the shared buffer.

## Selecting it

```sh
sb-cli init --benchmark_name gemm --flow transformed --backend esp --stage object \
    --instrumentation esp-ir --output_dir esp_ir_gemm
make -C benches/experiments/esp_ir_gemm
```

`--backend esp` alone selects the `esp` recipe; the explicit `--instrumentation`
wins. The same runtime serves both: under the bare-pointer convention a rank-0
memref is a single pointer, so `memref<i32>` and the `i64` handle are the same
C function.

## What it needs

Beyond the plugin: a soda-opt that registers `fold-memref-alias-ops` and
`affine-loop-normalize`. Upstream soda-opt registers passes by hand and lacks
both (`mlir-opt` has them); the change is four lines in
`tools/soda-opt/soda-opt.cpp`. The [validation guide](../../../docs/ESPValidation.md)
includes an applicable companion patch, plugin-linking build flags, and complete
test commands.
