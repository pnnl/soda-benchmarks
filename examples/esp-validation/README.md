# ESP validation convenience entry point

For synthesis followed by ESP full-system simulation on emilia, start with the [step-by-step tutorial](../../docs/ESPSynthesisSimulation.md).

The regression tests live in `examples/soda-plugins/test/sodap/ESP/` and run
through lit, including the `mlir-cpu-runner` ABI test and native profiler test.
The convenience entry point does not maintain a second set of test commands or
access hardware.

After configuring and building the plugin:

```sh
LLVM_LIT=/path/to/llvm-lit bash examples/esp-validation/run-tests.sh
```

`SODAP_BUILD_DIR` overrides the plugin build directory. `LLVM_LIT` may be omitted
when `llvm-lit` or `lit` is on PATH. Remaining arguments are passed to lit.
The full repository test target remains `check-sodap`; CI need not use this helper.

See [ESPValidation.md](../../docs/ESPValidation.md) for prerequisites and the
distinction between compiler/mock checks and board numerical validation.

The separate [generated floating-point GEMM example](generated-gemm/README.md)
contains the ESP adapter, application and replay scripts used to validate
supplied host LLVM IR and Bambu RTL, including the raw-f32 runtime API.
