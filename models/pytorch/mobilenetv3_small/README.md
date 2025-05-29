# PyTorch to LLVM IR Lowering Example

In this example, we will show how to lower a MobileNetV3_small PyTorch model to LLVM IR
using scripts from the MLIR Tools used by SODA [docker image](https://hub.docker.com/r/agostini01/soda).


## Instructions for docker users

1. Install docker and vscode.
2. Open the the `soda-benchmarks` project in a vscode development container. 
`cmd+shift+p` and select : `Dev Containers: Reopen in Container` this will download the
docker image and start the container. 
3. Once you container, you can enter this folder and run `make` to compile the tutorial.


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