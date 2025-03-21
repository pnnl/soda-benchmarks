# PyTorch to LLVM IR Lowering Example

In this example, we will show how to lower a simple PyTorch model to LLVM IR
using scripts from the MLIR Tools used by SODA [docker image](https://hub.docker.com/r/agostini01/mlir-tools-bookworm).

## Approach

Currently annotating `.dot` files with new attributes to represent features.
See [dot language spec](https://graphviz.org/doc/info/lang.html) for more
information on the sintax.

The annotated dot include a new attribute with a string that includes a dictionary `attr_name = "{id = val, ...}"`. This string has to be parsed into a dictionary.

Vertix example:

```
v11 [fillcolor = "0.000000 1.0 1.0", label = "tosa.matmul : (tensor<1x8x12xf...)\n", shape = ellipse, style = filled, 
     features = "{numArithmeticOpsEstimative = 3072, numMemoryOpsEstimative = 6144}"];
```

Edge example

```
v9 -> v11 [label = "0", style = solid, features = "{numElements = 128, bytesPerElement = 4}"];
```

in the future we can also include a costs attribute that will follow the format for either edges or vertices:

```
costs = "{swCost = 0.0, hwCost = 0.0"}"

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