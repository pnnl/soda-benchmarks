# PyTorch to LLVM IR Lowering Example

In this example, we will show how to lower a simple PyTorch model to LLVM IR
using scripts from the MLIR Tools used by SODA [docker image](https://hub.docker.com/r/agostini01/mlir-tools-bookworm).


## Instructions for docker users

This will mount the current folder into the docker container. Once inside the
container, you can run `make` to compile the tutorial.


## Artifacts

```
docker-version/
└── output
    ├── 01_tosa.mlir
    ├── 02_linalg.mlir
    ├── 02_linalg_on_tensors.mlir
    ├── 03_llvm.mlir
    └── 04_llvm.ll
```