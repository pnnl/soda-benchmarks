#!/usr/bin/env python3
"""
PolyBench atax Kernel: Matrix Transpose and Vector Multiply

Reference: PolyBenchC-4.2.1/linear-algebra/kernels/atax/atax.c

Mathematical Operation:
    y := A^T * (A * x)

where:
    - tmp = A * x  (matrix-vector product)
    - y = A^T * tmp  (transpose matrix-vector product)

This module implements the atax kernel as a PyTorch nn.Module and provides
MLIR generation capability via command-line interface.

Usage:
    # Direct execution for MLIR generation
    python kernel_atax.py ./output/atax_linalg.mlir --dialect linalg-on-tensors

    # Import as module
    from kernel_atax import Atax, init_array
    model = Atax(m=1900, n=2100)
    A, x = init_array(1900, 2100)
    result = model(A, x)
"""

import argparse

import torch
import torch.nn as nn

try:
    from PolyBenchPyTorch.linear_algebra.utils import (
        generate_mlir,
        make_parser,
        resolve_dtype,
    )
    from PolyBenchPyTorch.linear_algebra.utils import (
        get_dataset_dimensions as _get_dataset_dimensions,
    )
except ImportError:
    import importlib.util
    import os as _os

    _p = _os.path.abspath(
        _os.path.join(_os.path.dirname(__file__), "..", "..", "utils.py")
    )
    _spec = importlib.util.spec_from_file_location("la_utils", _p)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    make_parser = _mod.make_parser
    _get_dataset_dimensions = _mod.get_dataset_dimensions
    resolve_dtype = _mod.resolve_dtype
    generate_mlir = _mod.generate_mlir


def get_dataset_dimensions(dataset: str) -> dict:
    return _get_dataset_dimensions("atax", dataset)


class Atax(nn.Module):
    """
    PolyBench atax kernel: y := A^T * (A * x)

    Matrix transpose and vector multiply.

    Reference:
        PolyBenchC-4.2.1/linear-algebra/kernels/atax/atax.c

    Args:
        m: Rows in A (default: 1900)
        n: Columns in A (default: 2100)
    """

    def __init__(self, m: int, n: int) -> None:
        super().__init__()
        self.m = m
        self.n = n

    def forward(
        self,
        A: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Execute atax kernel computation.

        Adaptation from C: Nested loops converted to PyTorch matrix operations.
        C version computes tmp[i] = sum_j(A[i][j] * x[j]),
        then y[j] = sum_i(A[i][j] * tmp[i]).

        Args:
            A: Input matrix (m, n)
            x: Input vector (n,)

        Returns:
            y: Output vector (n,) = A^T * (A * x)

        Raises:
            RuntimeError: If tensor shapes are incompatible
        """
        # Shape assertions
        if not torch.jit.is_tracing():
            assert A.shape == (self.m, self.n), (
                f"A shape mismatch: {A.shape} != ({self.m}, {self.n})"
            )
            assert x.shape == (self.n,), f"x shape mismatch: {x.shape} != ({self.n},)"

        # Computation: y := A^T * (A * x)
        tmp = torch.matmul(A, x)  # (m,)
        y = torch.matmul(A.T, tmp)  # (n,)

        return y


def init_array(
    m: int, n: int, dtype: torch.dtype = torch.float32
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Initialize arrays for atax kernel matching C reference implementation.

    Formulas from PolyBenchC-4.2.1/linear-algebra/kernels/atax/atax.c lines 39-45:
        x[i] = 1 + (i / n)
        A[i, j] = ((i+j) % n) / (5*m)

    Args:
        m: Dimension parameter
        n: Dimension parameter
        dtype: Tensor data type (default: torch.float32)

    Returns:
        Tuple of (A, x):
            A (Tensor): (m, n) initialized matrix
            x (Tensor): (n,) initialized vector
    """
    # Initialize x (n,)
    x = torch.zeros(n, dtype=dtype)
    for i in range(n):
        x[i] = 1 + (i / n)

    # Initialize A (m x n)
    A = torch.zeros((m, n), dtype=dtype)
    for i in range(m):
        for j in range(n):
            A[i, j] = ((i + j) % n) / (5 * m)

    return A, x


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for MLIR generation."""
    parser = make_parser("atax", "./output/atax_linalg.mlir")
    parser.add_argument("--m", type=int, help="Dimension m (overrides dataset value)")
    parser.add_argument("--n", type=int, help="Dimension n (overrides dataset value)")
    return parser.parse_args()


def main() -> None:
    """Generate MLIR from atax kernel model."""
    args = parse_args()

    dims = get_dataset_dimensions(args.dataset)
    m = args.m if args.m is not None else dims["m"]
    n = args.n if args.n is not None else dims["n"]

    dtype = resolve_dtype(args.dtype)

    model = Atax(m, n)
    A, x = init_array(m, n, dtype=dtype)

    print(f"Compiling atax kernel to MLIR dialect: {args.dialect}")
    generate_mlir(model, (A, x), args.out_mlir_path, args.dialect)


if __name__ == "__main__":
    main()
