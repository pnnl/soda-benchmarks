# Instrumentation

Here you will learn how to add custom IPs and trigger their instantiation using MLIR.

## Using the `sb-cli` to trigger instrumentation and IP integration flows

Instrumentation flow depends on the transform dialect recipes and are paired with passes compiled from examples/soda-plugins. Consequently, you should follow the instructions to compile soda-plugins which puts the compiled plugin at: soda-benchmarks/examples/soda-plugins/build/lib/SODAPlugin.so and is by default detected by the templated Makefiles.

To scaffold an example with the instrumentation flow, you can use the `--instrumentation` option with the `sb-cli init` command. For example:

```bash
pixi run sb-cli init \
    --benchmark_name PolyBenchPyTorch.linear_algebra.blas.gemm \
    --dataset MINI \
    --dtype float32 \
    --device nangate45 \
    --clock_period 5 \
    --flow transformed \
    --backend bambu \
    --stage simulation \
    --instrumentation assert
```

Enter the directory printed by this command and run `pixi run make` to generate the Verilog and run the simulation. The generated Verilog will include the assert IP and the simulation will print the assert messages.