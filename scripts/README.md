# Conversion and Lowering Scripts

This directory contains scripts that are used to convert the input,
typically a model in a high-level language, into MLIR or lower abstraction.

## Model Conversion

The following scripts are used to translate models from high-level frameworks
to TOSA MLIR.

* [tflite_to_tosa.sh](tflite_to_tosa.sh) - Converts a TFLite model `<model.tflite> to TOSA MLIR.
* [protobuf_to_tosa.sh](protobuf_to_tosa.sh) - Converts a protobuf model `<model.pb>` or `<model.pbtxt> to TOSA MLIR.
* [tf_to_tosa.sh](tf_to_tosa.sh) - Converts a TensorFlow model `<model.pb>` or `<model.pbtxt>` to TOSA MLIR.

## Lowering

The following scripts should be used in the given order to lower the TOSA MLIR to LLVM IR.

1. [tosa_to_linalg.sh](tosa_to_linalg.sh) - Lowers TOSA MLIR to Linalg dialect.
2. [linalg_to_llvm.sh](linalg_to_llvm.sh) - Lowers Linalg dialect to LLVM dialect.
3. [llvm_to_ll.sh](llvm_to_ll.sh) - Translates MLIR with LLVM dialect to LLVM IR.

Typically, the lowering scripts will generate the following files:

* `01_tosa.mlir` - TOSA MLIR
* `02_linalg.mlir` - Linalg dialect using buffers
* `03_llvm.mlir` - LLVM dialect using buffers
* `04_llvm.ll` - LLVM IR file

## Backends

The scripts that consume the LLVM IR produced above, one per `sb-cli --backend`.
See [docs/ESPBackend.md](../docs/ESPBackend.md) for the cpu and esp ones.

* [ll_to_verilog.sh](ll_to_verilog.sh) - Synthesizes `<input.ll>` with Bambu (`--backend bambu`).
* [ll_to_binary.sh](ll_to_binary.sh) - Links `<input.ll>` into a native executable (`--backend cpu`).
* [ll_to_riscv.sh](ll_to_riscv.sh) - Cross-compiles `<input.ll>` for an ESP SoC's RISC-V core and stages a baremetal application (`--backend esp`).
* [link_esp_app.sh](link_esp_app.sh) - Finishes that application's link inside an ESP checkout.
* [kernel_arg_order.sh](kernel_arg_order.sh) - Prints the order `-soda-outline-bambu-code` gave `forward_kernel`'s parameters, which is not the order `@forward` declares them in.
* [esp_gemm_demo.sh](esp_gemm_demo.sh) - Runs all three of the above configurations on PolyBench gemm.

## Using the templates

The folder [templates/make/](templates/make/) contains Makefile templates to translate the model and convert it to LLVM IR. To use the templates, copy the correct template to the root of the project and rename it to `Makefile`. Adjust the variables in the template to match your project.
Then, run `make` to translate the model and lower to LLVM IR.

Similarly, the folder [templates/bash/](templates/bash/) contains bash scripts to translate the model and convert it to LLVM IR. To use the templates, copy the correct template to the root of the project and rename it to `compile.sh`. Adjust the variables in the template to match your project.
Then, run `./compile.sh` to translate the model and lower to LLVM IR.
