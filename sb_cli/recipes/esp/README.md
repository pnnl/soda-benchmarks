# ESP accelerator offload

Unlike the `assert` / `hw-counters` / `change-location` recipes, this one does not
*observe* the kernel — like `vector-dot`, it *replaces* a matched operation with a
call to hardware. Every `linalg.batch_matmul` becomes the ESP invocation sequence:

```
esp_alloc_shared -> esp_float2fixed_f32 (A) -> esp_float2fixed_f32 (B)
  -> esp_accel_cfg_regs -> esp_accel_start -> esp_accel_wait
  -> esp_fixed2float_f32 (C) -> esp_free_shared
```

The seven symbols are declared in
`examples/soda-plugins/include/sodap/ExecutionEngine/ESPRuntime.h` and have two
implementations: `EspRuntimeMock.cpp` (prints each call; what the `cpu` backend
links) and `EspRuntime.cpp` (drives the real FFN accelerator; built only inside an
ESP checkout).

## No `IPs/`

There is nothing for Bambu to integrate: the accelerator is a device on the ESP
SoC, reached over its memory-mapped registers by a runtime the kernel links
against. The recipe is a schedule and nothing else, and
`sb_cli.flow.ip_integration_block` returns an empty string for it.

## What it needs

`SODAPlugin.so` has to be built, because `sodap-linalg-batch-matmul-to-esp` lives
in it:

```sh
cmake --build examples/soda-plugins/build --target SODAPlugin
```

`scripts/soda_to_llvm_transformed.sh` picks it up from
`examples/soda-plugins/build/lib/SODAPlugin.so` by default; `MLIR_PLUGIN_PATH`
overrides that. Without the plugin, soda-opt silently runs the schedule against a
pass it does not know and the build fails inside the transform interpreter.

## Which kernels it matches

`linalg.batch_matmul` with static shapes. PolyBench kernels reach it through TOSA,
which always emits a batch dimension — gemm MINI lowers to
`linalg.batch_matmul ins(memref<1x20x30xf32>, memref<1x30x25xf32>) outs(memref<1x20x25xf32>)`,
i.e. batch 1, M 20, K 30, N 25. Only the matmul is offloaded; the `alpha` and
`beta` scalings stay in software as `linalg.generic` ops.

## Selecting it

```sh
sb-cli init --benchmark_name gemm --flow transformed --backend esp --stage object
```

`--backend esp` selects this recipe on its own. To get the ESP lowering while
building for the host — which is how the call sequence and the runtime ABI get
checked without hardware — ask for it explicitly:

```sh
sb-cli init --benchmark_name gemm --flow transformed \
    --backend cpu --stage simulation --instrumentation esp
```
