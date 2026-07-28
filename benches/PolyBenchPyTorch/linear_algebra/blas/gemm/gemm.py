#!/usr/bin/env python3
"""PolyBench GEMM Kernel: General Matrix Multiply

Reference: PolyBenchC-4.2.1/linear-algebra/blas/gemm/gemm.c

Operation:
    C := alpha * A * B + beta * C

This module implements the GEMM kernel as a PyTorch nn.Module and provides
MLIR generation capability via command-line interface.
"""

import argparse

import torch
import torch.nn as nn

from benches.PolyBenchPyTorch.linear_algebra.utils import (
    generate_mlir,
    get_dataset_dimensions,
    make_parser,
    resolve_dtype,
)


class Gemm(nn.Module):
    """GEMM kernel: C := alpha*A*B + beta*C

    Args:
        ni: Rows in A, C
        nj: Columns in B, C
        nk: Columns in A, rows in B
    """

    def __init__(self, ni: int, nj: int, nk: int) -> None:
        super().__init__()
        self.ni = ni
        self.nj = nj
        self.nk = nk

    def forward(
        self,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        # Shape assertions
        if not torch.jit.is_tracing():
            assert A.shape == (self.ni, self.nk), (
                f"A shape mismatch: {A.shape} != ({self.ni}, {self.nk})"
            )
            assert B.shape == (self.nk, self.nj), (
                f"B shape mismatch: {B.shape} != ({self.nk}, {self.nj})"
            )
            assert C.shape == (self.ni, self.nj), (
                f"C shape mismatch: {C.shape} != ({self.ni}, {self.nj})"
            )

        # Computation
        tmp = alpha * torch.matmul(A, B)  # (ni, nj)
        C_out = beta * C + tmp

        return C_out


def init_array(
    ni: int, nj: int, nk: int, dtype: torch.dtype = torch.float32
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Initialize arrays for GEMM following data-model.md formulas.

    Returns (alpha, beta, A, B, C)
    alpha and beta are 0-dim torch tensors as required for torch-mlir.
    """
    alpha = torch.tensor(1.5, dtype=dtype)
    beta = torch.tensor(1.2, dtype=dtype)

    C = torch.zeros((ni, nj), dtype=dtype)
    for i in range(ni):
        for j in range(nj):
            C[i, j] = ((i * j + 1) % ni) / float(ni)

    A = torch.zeros((ni, nk), dtype=dtype)
    for i in range(ni):
        for j in range(nk):
            A[i, j] = ((i * (j + 1)) % nk) / float(nk)

    B = torch.zeros((nk, nj), dtype=dtype)
    for i in range(nk):
        for j in range(nj):
            B[i, j] = ((i * (j + 2)) % nj) / float(nj)

    return alpha, beta, A, B, C


def parse_args() -> argparse.Namespace:
    parser = make_parser("gemm", "./output/gemm_linalg.mlir")
    parser.add_argument("--ni", type=int, help="Dimension ni (overrides dataset value)")
    parser.add_argument("--nj", type=int, help="Dimension nj (overrides dataset value)")
    parser.add_argument("--nk", type=int, help="Dimension nk (overrides dataset value)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dims = get_dataset_dimensions("gemm", args.dataset)
    ni = args.ni if args.ni is not None else dims["ni"]
    nj = args.nj if args.nj is not None else dims["nj"]
    nk = args.nk if args.nk is not None else dims["nk"]

    dtype = resolve_dtype(args.dtype)

    model = Gemm(ni, nj, nk)
    alpha, beta, A, B, C = init_array(ni, nj, nk, dtype=dtype)

    print(f"Compiling GEMM kernel to MLIR dialect: {args.dialect}")
    generate_mlir(model, (alpha, beta, A, B, C), args.out_mlir_path, args.dialect)


if __name__ == "__main__":
    main()
