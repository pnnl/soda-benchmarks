# addr_gen_testing

This folder contains focused lit tests for the `gen-addr-function-pass` in `SODAPlugin`.

## Test files

- `run_addresses.mlir`
  - End-to-end runtime check for base-0 address traces (`base0=`) across dense, strided, and permuted access patterns.
  - Uses `mlir-cpu-runner` plus `FileCheck`.

- `addr_gen_multi_layouts.mlir`
  - Static IR check that helper generation covers 1D/2D/3D memrefs, strided layouts, and dedup behavior.
  - Verifies expected helper count/patterns with `CHECK-COUNT` and `CHECK-NOT`.

- `addr_gen_matmul_edge.mlir`
  - Static IR check around matmul/broadcast edge patterns and map-sensitive helper generation.

- `addr_gen_matmul_2x3_3x2_static.mlir`
  - Small, explicit 2x3 x 3x2 matmul case used to check address generation of a matmul operation.

## Running only this folder

From the plugin build directory:

```sh
cd /workspaces/soda-benchmarks/examples/soda-plugins/build/test
/home/developer/.local/bin/lit -sv . --filter=addr_gen
```

If `check-sodap` is configured with a valid lit path, you can also run:

```sh
cd /workspaces/soda-benchmarks/examples/soda-plugins/build
cmake --build . --target check-sodap
```
