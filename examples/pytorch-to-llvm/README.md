# PyTorch to LLVM IR Lowering Example

In this example, we will show how to lower a simple PyTorch model to LLVM IR
using scripts from the MLIR Tools used by SODA [docker image](https://hub.docker.com/r/agostini01/soda).


## Instructions for Docker Users

1. Install Docker, VS Code, and the VS Code [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-azuretools.vscode-containers).
2. Open the `soda-benchmarks` project in a VS Code development container. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on macOS) and select: `Dev Containers: Reopen in Container`. This will download the Docker image and start the container.
3. Once inside the container, navigate to this folder and run `make` to compile the example.


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