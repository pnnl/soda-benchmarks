# ESP validation convenience entry point

The regression tests live in `examples/soda-plugins/test/sodap/ESP/` and run
through lit, including the `mlir-cpu-runner` ABI test and native profiler test.
This example does not maintain a second set of test commands or access hardware.

After configuring and building the plugin:

```sh
LLVM_LIT=/path/to/llvm-lit bash examples/esp-validation/run-tests.sh
```

`SODAP_BUILD_DIR` overrides the plugin build directory. `LLVM_LIT` may be omitted
when `llvm-lit` or `lit` is on PATH. Remaining arguments are passed to lit.
The full repository test target remains `check-sodap`; CI need not use this helper.

See [ESPValidation.md](../../docs/ESPValidation.md) for prerequisites and the
distinction between compiler/mock checks and board numerical validation.
