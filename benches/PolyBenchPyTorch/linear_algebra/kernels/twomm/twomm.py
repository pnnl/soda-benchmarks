#!/usr/bin/env python3
"""
PolyBench twomm Kernel: Double Matrix Multiplication

Reference: PolyBenchC-4.2.1/linear-algebra/kernels/2mm/2mm.c

Mathematical Operation:
    D := alpha * A * B * C + beta * D

where:
    - tmp = alpha * A * B  (intermediate result)
    - D_out = beta * D + tmp * C  (final result)

This module implements the twomm kernel as a PyTorch nn.Module and provides
MLIR generation capability via command-line interface.

Usage:
    # Direct execution for MLIR generation
    python kernel_2mm.py ./output/2mm_linalg.mlir --dialect linalg-on-tensors

    # Import as module
    from kernel_2mm import TwoMM, init_array
    model = TwoMM(ni=800, nj=900, nk=1100, nl=1200)
    alpha, beta, A, B, C, D = init_array(800, 900, 1100, 1200)
    result = model(alpha, beta, A, B, C, D)
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
    return _get_dataset_dimensions("twomm", dataset)


class TwoMM(nn.Module):
    """
    PolyBench twomm kernel: D := alpha*A*B*C + beta*D

    Double matrix multiplication with scalar coefficients.

    Reference:
        PolyBenchC-4.2.1/linear-algebra/kernels/2mm/2mm.c

    Args:
        ni: Rows in A, D, tmp (default: 800)
        nj: Columns in B, tmp; rows in C (default: 900)
        nk: Columns in A, rows in B (default: 1100)
        nl: Columns in C, D (default: 1200)
    """

    def __init__(self, ni: int, nj: int, nk: int, nl: int) -> None:
        super().__init__()
        self.ni = ni
        self.nj = nj
        self.nk = nk
        self.nl = nl

    def forward(
        self,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
    ) -> torch.Tensor:
        """
        Execute twomm kernel computation.

        Adaptation from C: Nested loops converted to PyTorch matrix operations.
        C version computes tmp[i][j] = alpha * sum_k(A[i][k] * B[k][j]),
        then D[i][l] = beta*D[i][l] + sum_j(tmp[i][j] * C[j][l]).

        Args:
            alpha: Scalar coefficient for first multiplication
            beta: Scalar coefficient for D scaling
            A: Input matrix (ni, nk)
            B: Input matrix (nk, nj)
            C: Input matrix (nj, nl)
            D: Input matrix (ni, nl)

        Returns:
            D_out: Result matrix (ni, nl) = alpha*A*B*C + beta*D

        Raises:
            RuntimeError: If tensor shapes are incompatible
        """
        # Shape assertions
        if not torch.jit.is_tracing():
            assert A.shape == (self.ni, self.nk), (
                f"A shape mismatch: {A.shape} != ({self.ni}, {self.nk})"
            )
            assert B.shape == (self.nk, self.nj), (
                f"B shape mismatch: {B.shape} != ({self.nk}, {self.nj})"
            )
            assert C.shape == (self.nj, self.nl), (
                f"C shape mismatch: {C.shape} != ({self.nj}, {self.nl})"
            )
            assert D.shape == (self.ni, self.nl), (
                f"D shape mismatch: {D.shape} != ({self.ni}, {self.nl})"
            )

        # Computation: D := alpha * A * B * C + beta * D
        tmp = alpha * torch.matmul(A, B)  # (ni, nj)
        D_out = beta * D + torch.matmul(tmp, C)  # (ni, nl)

        return D_out


def init_array(
    ni: int, nj: int, nk: int, nl: int, dtype: torch.dtype = torch.float32
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """
    Initialize arrays for twomm kernel matching C reference implementation.

    Formulas from PolyBenchC-4.2.1/linear-algebra/kernels/2mm/2mm.c lines 36-49:
        alpha = 1.5
        beta = 1.2
        A[i, j] = ((i*j+1) % ni) / ni
        B[i, j] = (i*(j+1) % nj) / nj
        C[i, j] = ((i*(j+3)+1) % nl) / nl
        D[i, j] = (i*(j+2) % nk) / nk

    Args:
        ni: Dimension parameter
        nj: Dimension parameter
        nk: Dimension parameter
        nl: Dimension parameter
        dtype: Tensor data type (default: torch.float32 for C float equivalence)

    Returns:
        Tuple of (alpha, beta, A, B, C, D):
            alpha (float): 1.5
            beta (float): 1.2
            A (Tensor): (ni, nk) initialized matrix
            B (Tensor): (nk, nj) initialized matrix
            C (Tensor): (nj, nl) initialized matrix
            D (Tensor): (ni, nl) initialized matrix
    """
    # Return alpha and beta as 0-dim torch tensors so torch-mlir accepts them
    # as example args. Use the same dtype as the initialized arrays.
    alpha = torch.tensor(1.5, dtype=dtype)
    beta = torch.tensor(1.2, dtype=dtype)

    # Initialize A (ni x nk)
    A = torch.zeros((ni, nk), dtype=dtype)
    for i in range(ni):
        for j in range(nk):
            A[i, j] = ((i * j + 1) % ni) / ni

    # Initialize B (nk x nj)
    B = torch.zeros((nk, nj), dtype=dtype)
    for i in range(nk):
        for j in range(nj):
            B[i, j] = (i * (j + 1) % nj) / nj

    # Initialize C (nj x nl)
    C = torch.zeros((nj, nl), dtype=dtype)
    for i in range(nj):
        for j in range(nl):
            C[i, j] = ((i * (j + 3) + 1) % nl) / nl

    # Initialize D (ni x nl)
    D = torch.zeros((ni, nl), dtype=dtype)
    for i in range(ni):
        for j in range(nl):
            D[i, j] = (i * (j + 2) % nk) / nk

    return alpha, beta, A, B, C, D


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for MLIR generation."""
    parser = make_parser("twomm", "./output/2mm_linalg.mlir")
    parser.add_argument("--ni", type=int, help="Dimension ni (overrides dataset value)")
    parser.add_argument("--nj", type=int, help="Dimension nj (overrides dataset value)")
    parser.add_argument("--nk", type=int, help="Dimension nk (overrides dataset value)")
    parser.add_argument("--nl", type=int, help="Dimension nl (overrides dataset value)")
    return parser.parse_args()


def main() -> None:
    """Generate MLIR from twomm kernel model."""
    args = parse_args()

    dims = get_dataset_dimensions(args.dataset)
    ni = args.ni if args.ni is not None else dims["ni"]
    nj = args.nj if args.nj is not None else dims["nj"]
    nk = args.nk if args.nk is not None else dims["nk"]
    nl = args.nl if args.nl is not None else dims["nl"]

    dtype = resolve_dtype(args.dtype)

    model = TwoMM(ni, nj, nk, nl)
    alpha, beta, A, B, C, D = init_array(ni, nj, nk, nl, dtype=dtype)

    print(f"Compiling twomm kernel to MLIR dialect: {args.dialect}")
    generate_mlir(model, (alpha, beta, A, B, C, D), args.out_mlir_path, args.dialect)


if __name__ == "__main__":
    main()
