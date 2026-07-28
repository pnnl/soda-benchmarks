# PolyBench BLAS Kernels (PyTorch)

PyTorch implementations of PolyBench/C linear algebra BLAS kernels with MLIR generation support.

## Kernels

| Kernel   | Operation                                      |
|----------|------------------------------------------------|
| gemm     | C := alpha*A*B + beta*C                        |
| gemver   | Multi-step rank-2 update + matrix-vector ops   |
| gesummv  | z := alpha*A*x + beta*B*x                      |
| symm     | C := alpha*A*B + beta*C (A symmetric)          |
| syr2k    | C := alpha*(A*B^T + B*A^T) + beta*C            |
| syrk     | C := alpha*A*A^T + beta*C                      |
| trmm     | B := alpha*A^T*B (A lower triangular, unit diag)|

## Running Tests

From the `benches/` directory:

```bash
python test_all_kernels.py
```

## Generating MLIR

Each kernel can be run directly to produce MLIR output:

```bash
# Default: TOSA dialect, MINI dataset, float32
pixi run python -m benches.PolyBenchPyTorch.linear_algebra.blas.gemm.gemm output/gemm.mlir

# With options
pixi run python -m benches.PolyBenchPyTorch.linear_algebra.blas.gemm.gemm output/gemm.mlir \
    --dialect tosa --dataset SMALL --dtype float32
```

Available dialects: `tosa`, `linalg-on-tensors`, `torch`, `raw`, `mhlo`

Available datasets: `MINI`, `SMALL`, `MEDIUM`, `LARGE`, `EXTRALARGE`

## Requirements

- Python 3.11
- PyTorch 2.4+
- torch-mlir (for MLIR generation)
