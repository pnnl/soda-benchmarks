#!/usr/bin/env python3
"""
PolyBench threemm Kernel: Triple Matrix Multiplication

Reference: PolyBenchC-4.2.1/linear-algebra/kernels/3mm/3mm.c

Mathematical Operation:
    G := (A * B) * (C * D)

where:
    - E = A * B  (first intermediate)
    - F = C * D  (second intermediate)
    - G = E * F  (final result)

This module implements the threemm kernel as a PyTorch nn.Module and provides
MLIR generation capability via command-line interface.

Usage:
    # Direct execution for MLIR generation
    python kernel_3mm.py ./output/3mm_linalg.mlir --dialect linalg-on-tensors

    # Import as module
    from kernel_3mm import ThreeMM, init_array
    model = ThreeMM(ni=800, nj=900, nk=1000, nl=1100, nm=1200)
    A, B, C, D = init_array(800, 900, 1000, 1100, 1200)
    result = model(A, B, C, D)
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
    return _get_dataset_dimensions("threemm", dataset)


class ThreeMM(nn.Module):
    """
    PolyBench threemm kernel: G := (A*B) * (C*D)

    Triple matrix multiplication.

    Reference:
        PolyBenchC-4.2.1/linear-algebra/kernels/3mm/3mm.c

    Args:
        ni: Rows in A, E, G (default: 800)
        nj: Columns in B, E; rows in C, F (default: 900)
        nk: Columns in A, rows in B (default: 1000)
        nl: Columns in D, F, G (default: 1100)
        nm: Columns in C, rows in D (default: 1200)
    """

    def __init__(self, ni: int, nj: int, nk: int, nl: int, nm: int) -> None:
        super().__init__()
        self.ni = ni
        self.nj = nj
        self.nk = nk
        self.nl = nl
        self.nm = nm

    def forward(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
    ) -> torch.Tensor:
        """
        Execute threemm kernel computation.

        Adaptation from C: Nested loops converted to PyTorch matrix operations.
        C version computes E[i][j] = sum_k(A[i][k] * B[k][j]),
        F[j][l] = sum_m(C[j][m] * D[m][l]),
        G[i][l] = sum_j(E[i][j] * F[j][l]).

        Args:
            A: Input matrix (ni, nk)
            B: Input matrix (nk, nj)
            C: Input matrix (nj, nm)
            D: Input matrix (nm, nl)

        Returns:
            G: Result matrix (ni, nl) = (A*B) * (C*D)

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
            assert C.shape == (self.nj, self.nm), (
                f"C shape mismatch: {C.shape} != ({self.nj}, {self.nm})"
            )
            assert D.shape == (self.nm, self.nl), (
                f"D shape mismatch: {D.shape} != ({self.nm}, {self.nl})"
            )

        # Computation: G := (A * B) * (C * D)
        E = torch.matmul(A, B)  # (ni, nj)
        F = torch.matmul(C, D)  # (nj, nl)
        G = torch.matmul(E, F)  # (ni, nl)

        return G


def init_array(
    ni: int, nj: int, nk: int, nl: int, nm: int, dtype: torch.dtype = torch.float32
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Initialize arrays for threemm kernel matching C reference implementation.

    Formulas from PolyBenchC-4.2.1/linear-algebra/kernels/3mm/3mm.c lines 43-55:
        A[i, j] = ((i*j+1) % ni) / (5*ni)
        B[i, j] = ((i*(j+1)+1) % nj) / (5*nj)
        C[i, j] = ((i*(j+3)+1) % nl) / (5*nl)
        D[i, j] = ((i*(j+2)+1) % nk) / (5*nk)

    Args:
        ni: Dimension parameter
        nj: Dimension parameter
        nk: Dimension parameter
        nl: Dimension parameter
        nm: Dimension parameter
        dtype: Tensor data type (default: torch.float32)

    Returns:
        Tuple of (A, B, C, D):
            A (Tensor): (ni, nk) initialized matrix
            B (Tensor): (nk, nj) initialized matrix
            C (Tensor): (nj, nm) initialized matrix
            D (Tensor): (nm, nl) initialized matrix
    """
    # Initialize A (ni x nk)
    A = torch.zeros((ni, nk), dtype=dtype)
    for i in range(ni):
        for j in range(nk):
            A[i, j] = ((i * j + 1) % ni) / (5 * ni)

    # Initialize B (nk x nj)
    B = torch.zeros((nk, nj), dtype=dtype)
    for i in range(nk):
        for j in range(nj):
            B[i, j] = ((i * (j + 1) + 1) % nj) / (5 * nj)

    # Initialize C (nj x nm)
    C = torch.zeros((nj, nm), dtype=dtype)
    for i in range(nj):
        for j in range(nm):
            C[i, j] = ((i * (j + 3) + 1) % nl) / (5 * nl)

    # Initialize D (nm x nl)
    D = torch.zeros((nm, nl), dtype=dtype)
    for i in range(nm):
        for j in range(nl):
            D[i, j] = ((i * (j + 2) + 1) % nk) / (5 * nk)

    return A, B, C, D


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for MLIR generation."""
    parser = make_parser(
        "threemm",
        "./output/3mm_linalg.mlir",
        extra_dtype_choices=["int32", "int16", "int64"],
    )
    parser.add_argument("--ni", type=int, help="Dimension ni (overrides dataset value)")
    parser.add_argument("--nj", type=int, help="Dimension nj (overrides dataset value)")
    parser.add_argument("--nk", type=int, help="Dimension nk (overrides dataset value)")
    parser.add_argument("--nl", type=int, help="Dimension nl (overrides dataset value)")
    parser.add_argument("--nm", type=int, help="Dimension nm (overrides dataset value)")
    return parser.parse_args()


def main() -> None:
    """Generate MLIR from threemm kernel model."""
    args = parse_args()

    dims = get_dataset_dimensions(args.dataset)
    ni = args.ni if args.ni is not None else dims["ni"]
    nj = args.nj if args.nj is not None else dims["nj"]
    nk = args.nk if args.nk is not None else dims["nk"]
    nl = args.nl if args.nl is not None else dims["nl"]
    nm = args.nm if args.nm is not None else dims["nm"]

    dtype = resolve_dtype(args.dtype)

    model = ThreeMM(ni, nj, nk, nl, nm)
    A, B, C, D = init_array(ni, nj, nk, nl, nm, dtype=dtype)

    print(f"Compiling threemm kernel to MLIR dialect: {args.dialect}")
    generate_mlir(model, (A, B, C, D), args.out_mlir_path, args.dialect)


if __name__ == "__main__":
    main()
