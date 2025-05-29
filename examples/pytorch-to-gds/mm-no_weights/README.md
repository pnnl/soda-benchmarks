# PyTorch to Verilog Generation Example

In this example, we will show how to lower a simple PyTorch model to GDS files
using scripts and binaries from the MLIR Tools used by SODA 
[docker image](https://hub.docker.com/r/agostini01/soda).


## Instructions for docker users

1. Install docker and vscode.
2. Open the the `soda-benchmarks` project in a vscode development container. 
`cmd+shift+p` and select : `Dev Containers: Reopen in Container` this will download the
docker image and start the container. 
3. Once you container, you can enter this folder and run `make` to compile the tutorial.


## Selecting an optimization strategy

Change the `TARGET=` Variable in the [`Makefile`](Makefile) to select the
optimization strategy.


## Artifacts

The `<strategy>` can be either `baseline` or `optimized` and is governed by the
optimization pipeline in `soda-opt`.

```
└── output
    ├── 01_tosa.mlir
    ├── 02_linalg_on_tensors.mlir
    ├── 02_linalg.mlir // with buffers
    ├── 04_llvm_<strategy>.mlir
    ├── 05_llvm_<strategy>.ll // LLVM IR file
    └── bambu/<strategy>/06_verilog.v
```