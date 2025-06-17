# PyTorch to GDS Generation Example

This example demonstrates how to lower a simple PyTorch model to GDS files using scripts and binaries from the MLIR tools provided by the SODA [Docker image](https://hub.docker.com/r/agostini01/soda).


## Instructions for Docker Users

1. Install Docker, VS Code, and the VS Code [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-azuretools.vscode-containers).
2. Open the `soda-benchmarks` project in a VS Code development container. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on macOS) and select: `Dev Containers: Reopen in Container`. This will download the Docker image and start the container.
3. Once inside the container, navigate to this folder and run `make` to compile the example.


## Selecting an Optimization Strategy

Change the `TARGET=` variable in the [`Makefile`](Makefile) to select the desired optimization strategy.


## Artifacts

The `<strategy>` can be either `baseline` or `optimized`, and is determined by the optimization pipeline in `soda-opt`.

```
└── output
    ├── 01_tosa.mlir
    ├── 02_linalg_on_tensors.mlir
    ├── 02_linalg.mlir        // with buffers
    ├── 04_llvm_<strategy>.mlir
    ├── 05_llvm_<strategy>.ll // LLVM IR file
    └── bambu/<strategy>/06_verilog.v
```