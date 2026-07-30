#!/usr/bin/env python3
"""
PolyBench mvt Kernel: Matrix-Vector Product and Transpose

Reference: PolyBenchC-4.2.1/linear-algebra/kernels/mvt/mvt.c

Mathematical Operation:
    x1 := x1 + A * y1
    x2 := x2 + A^T * y2

Two independent accumulation operations.

This module implements the mvt kernel as a PyTorch nn.Module and provides
MLIR generation capability via command-line interface.

Usage:
    # Direct execution for MLIR generation
    python kernel_mvt.py ./output/mvt_linalg.mlir --dialect linalg-on-tensors

    # Import as module
    from kernel_mvt import Mvt, init_array
    model = Mvt(n=2000)
    inputs = init_array(2000)  # (x1, x2, y_1, y_2, A)
    x1_out, x2_out = model(*inputs)
"""

import argparse

import torch
import torch.nn as nn

from benches.PolyBenchPyTorch.linear_algebra.utils import (
    generate_mlir,
    make_parser,
    resolve_dtype,
)
from benches.PolyBenchPyTorch.linear_algebra.utils import (
    get_dataset_dimensions as _get_dataset_dimensions,
)


def get_dataset_dimensions(dataset: str) -> dict:
    return _get_dataset_dimensions("mvt", dataset)


class Mvt(nn.Module):
    """
    PolyBench mvt kernel: x1 := x1 + A*y1, x2 := x2 + A^T*y2

    Matrix-vector product and transpose with accumulation.

    Reference:
        PolyBenchC-4.2.1/linear-algebra/kernels/mvt/mvt.c

    Args:
        n: Size of square matrix A, length of all vectors (default: 2000)
    """

    def __init__(self, n: int) -> None:
        super().__init__()
        self.n = n

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        y_1: torch.Tensor,
        y_2: torch.Tensor,
        A: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Execute mvt kernel computation.

        Adaptation from C: Nested loops converted to PyTorch matrix operations.
        C version computes x1[i] = x1[i] + sum_j(A[i][j] * y_1[j]),
        and x2[i] = x2[i] + sum_j(A[j][i] * y_2[j]).

        Args:
            x1: Accumulation vector 1 (n,)
            x2: Accumulation vector 2 (n,)
            y_1: Input vector 1 (n,)
            y_2: Input vector 2 (n,)
            A: Input square matrix (n, n)

        Returns:
            Tuple of (x1_out, x2_out):
                x1_out: Updated x1 (n,) = x1 + A * y_1
                x2_out: Updated x2 (n,) = x2 + A^T * y_2

        Raises:
            RuntimeError: If tensor shapes are incompatible
        """
        # Shape assertions
        if not torch.jit.is_tracing():
            assert x1.shape == (self.n,), (
                f"x1 shape mismatch: {x1.shape} != ({self.n},)"
            )
            assert x2.shape == (self.n,), (
                f"x2 shape mismatch: {x2.shape} != ({self.n},)"
            )
            assert y_1.shape == (self.n,), (
                f"y_1 shape mismatch: {y_1.shape} != ({self.n},)"
            )
            assert y_2.shape == (self.n,), (
                f"y_2 shape mismatch: {y_2.shape} != ({self.n},)"
            )
            assert A.shape == (self.n, self.n), (
                f"A shape mismatch: {A.shape} != ({self.n}, {self.n})"
            )

        # Computation: x1 := x1 + A*y1, x2 := x2 + A^T*y2
        x1_out = x1 + torch.matmul(A, y_1)
        x2_out = x2 + torch.matmul(A.T, y_2)

        return x1_out, x2_out


def init_array(
    n: int, dtype: torch.dtype = torch.float32
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Initialize arrays for mvt kernel matching C reference implementation.

    Formulas from PolyBenchC-4.2.1/linear-algebra/kernels/mvt/mvt.c lines 46-56:
        x1[i] = 0
        x2[i] = 0
        y_1[i] = (i % n) / n
        y_2[i] = (i % n) / n
        A[i, j] = (i*j % n) / n

    Args:
        n: Dimension parameter
        dtype: Tensor data type (default: torch.float32)

    Returns:
        Tuple of (x1, x2, y_1, y_2, A), matching kernel_mvt() in the C reference:
            x1 (Tensor): (n,) initialized to zeros
            x2 (Tensor): (n,) initialized to zeros
            y_1 (Tensor): (n,) initialized vector
            y_2 (Tensor): (n,) initialized vector
            A (Tensor): (n, n) initialized square matrix
    """
    # Initialize x1, x2 (n,) - zeros
    x1 = torch.zeros(n, dtype=dtype)
    x2 = torch.zeros(n, dtype=dtype)

    # Initialize y_1, y_2 (n,)
    y_1 = torch.zeros(n, dtype=dtype)
    y_2 = torch.zeros(n, dtype=dtype)
    for i in range(n):
        y_1[i] = (i % n) / n
        y_2[i] = (i % n) / n

    # Initialize A (n x n)
    A = torch.zeros((n, n), dtype=dtype)
    for i in range(n):
        for j in range(n):
            A[i, j] = (i * j % n) / n

    return x1, x2, y_1, y_2, A


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for MLIR generation."""
    parser = make_parser("mvt", "./output/mvt_linalg.mlir")
    parser.add_argument("--n", type=int, help="Dimension n (overrides dataset value)")
    return parser.parse_args()


def main() -> None:
    """Generate MLIR from mvt kernel model."""
    args = parse_args()

    dims = get_dataset_dimensions(args.dataset)
    n = args.n if args.n is not None else dims["n"]

    dtype = resolve_dtype(args.dtype)

    model = Mvt(n)
    inputs = init_array(n, dtype=dtype)

    print(f"Compiling mvt kernel to MLIR dialect: {args.dialect}")
    generate_mlir(model, inputs, args.out_mlir_path, args.dialect)


if __name__ == "__main__":
    main()
