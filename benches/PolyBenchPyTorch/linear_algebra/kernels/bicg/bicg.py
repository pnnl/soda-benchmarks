#!/usr/bin/env python3
"""
PolyBench bicg Kernel: Bi-Conjugate Gradient

Reference: PolyBenchC-4.2.1/linear-algebra/kernels/bicg/bicg.c

Mathematical Operation:
    s := A^T * r
    q := A * p

Two independent operations with dual outputs.

This module implements the bicg kernel as a PyTorch nn.Module and provides
MLIR generation capability via command-line interface.

Usage:
    # Direct execution for MLIR generation
    python kernel_bicg.py ./output/bicg_linalg.mlir --dialect linalg-on-tensors

    # Import as module
    from kernel_bicg import Bicg, init_array
    model = Bicg(m=1900, n=2100)
    A, r, p = init_array(1900, 2100)
    s, q = model(A, r, p)
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
    return _get_dataset_dimensions("bicg", dataset)


class Bicg(nn.Module):
    """
    PolyBench bicg kernel: s := A^T * r, q := A * p

    Bi-conjugate gradient with dual outputs.

    Reference:
        PolyBenchC-4.2.1/linear-algebra/kernels/bicg/bicg.c

    Args:
        m: Columns in A, length of s, p (default: 1900)
        n: Rows in A, length of q, r (default: 2100)
    """

    def __init__(self, m: int, n: int) -> None:
        super().__init__()
        self.m = m
        self.n = n

    def forward(
        self,
        A: torch.Tensor,
        r: torch.Tensor,
        p: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Execute bicg kernel computation.

        Adaptation from C: Nested loops converted to PyTorch matrix operations.
        C version computes s[j] = sum_i(A[i][j] * r[i]),
        and q[i] = sum_j(A[i][j] * p[j]).

        Args:
            A: Input matrix (n, m)
            r: Input vector (n,)
            p: Input vector (m,)

        Returns:
            Tuple of (s, q):
                s: Output vector (m,) = A^T * r
                q: Output vector (n,) = A * p

        Raises:
            RuntimeError: If tensor shapes are incompatible
        """
        # Shape assertions
        if not torch.jit.is_tracing():
            assert A.shape == (self.n, self.m), (
                f"A shape mismatch: {A.shape} != ({self.n}, {self.m})"
            )
            assert r.shape == (self.n,), f"r shape mismatch: {r.shape} != ({self.n},)"
            assert p.shape == (self.m,), f"p shape mismatch: {p.shape} != ({self.m},)"

        # Computation: s := A^T * r, q := A * p
        s = torch.matmul(A.T, r)  # (m,)
        q = torch.matmul(A, p)  # (n,)

        return s, q


def init_array(
    m: int, n: int, dtype: torch.dtype = torch.float32
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Initialize arrays for bicg kernel matching C reference implementation.

    Formulas from PolyBenchC-4.2.1/linear-algebra/kernels/bicg/bicg.c lines 43-50:
        p[i] = (i % m) / m
        r[i] = (i % n) / n
        A[i, j] = (i*(j+1) % n) / n

    Args:
        m: Dimension parameter
        n: Dimension parameter
        dtype: Tensor data type (default: torch.float32)

    Returns:
        Tuple of (A, r, p):
            A (Tensor): (n, m) initialized matrix
            r (Tensor): (n,) initialized vector
            p (Tensor): (m,) initialized vector
    """
    # Initialize p (m,)
    p = torch.zeros(m, dtype=dtype)
    for i in range(m):
        p[i] = (i % m) / m

    # Initialize r (n,)
    r = torch.zeros(n, dtype=dtype)
    for i in range(n):
        r[i] = (i % n) / n

    # Initialize A (n x m)
    A = torch.zeros((n, m), dtype=dtype)
    for i in range(n):
        for j in range(m):
            A[i, j] = (i * (j + 1) % n) / n

    return A, r, p


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for MLIR generation."""
    parser = make_parser("bicg", "./output/bicg_linalg.mlir")
    parser.add_argument("--m", type=int, help="Dimension m (overrides dataset value)")
    parser.add_argument("--n", type=int, help="Dimension n (overrides dataset value)")
    return parser.parse_args()


def main() -> None:
    """Generate MLIR from bicg kernel model."""
    args = parse_args()

    dims = get_dataset_dimensions(args.dataset)
    m = args.m if args.m is not None else dims["m"]
    n = args.n if args.n is not None else dims["n"]

    dtype = resolve_dtype(args.dtype)

    model = Bicg(m, n)
    A, r, p = init_array(m, n, dtype=dtype)

    print(f"Compiling bicg kernel to MLIR dialect: {args.dialect}")
    generate_mlir(model, (A, r, p), args.out_mlir_path, args.dialect)


if __name__ == "__main__":
    main()
