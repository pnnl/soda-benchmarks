#!/usr/bin/env python3
"""PolyBench SYRK Kernel: C := alpha*A*A^T + beta*C"""

import argparse

import torch
import torch.nn as nn

from benches.PolyBenchPyTorch.linear_algebra.utils import (
    generate_mlir,
    get_dataset_dimensions,
    make_parser,
    resolve_dtype,
)


class Syrk(nn.Module):
    def __init__(self, n: int, m: int) -> None:
        super().__init__()
        self.n = n
        self.m = m

    def forward(
        self, alpha: torch.Tensor, beta: torch.Tensor, C: torch.Tensor, A: torch.Tensor
    ) -> torch.Tensor:
        if not torch.jit.is_tracing():
            assert A.shape == (self.n, self.m)
            assert C.shape == (self.n, self.n)

        result = alpha * torch.matmul(A, A.t()) + beta * C
        # Result is symmetric by construction; return full matrix (TOSA-compatible)
        return result


def init_array(n: int, m: int, dtype: torch.dtype = torch.float32):
    """Returns (alpha, beta, C, A)."""
    alpha = torch.tensor(1.5, dtype=dtype)
    beta = torch.tensor(1.2, dtype=dtype)
    C = torch.zeros((n, n), dtype=dtype)
    A = torch.zeros((n, m), dtype=dtype)

    for i in range(n):
        for j in range(n):
            C[i, j] = ((i * j + 2) % m) / float(m)
        for j in range(m):
            A[i, j] = ((i * j + 1) % n) / float(n)

    return alpha, beta, C, A


def parse_args() -> argparse.Namespace:
    parser = make_parser("syrk", "./output/syrk_linalg.mlir")
    parser.add_argument("--n", type=int, help="Dimension n (overrides dataset)")
    parser.add_argument("--m", type=int, help="Dimension m (overrides dataset)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dims = get_dataset_dimensions("syrk", args.dataset)
    n = args.n if args.n is not None else dims["n"]
    m = args.m if args.m is not None else dims["m"]

    dtype = resolve_dtype(args.dtype)

    model = Syrk(n, m)
    inputs = init_array(n, m, dtype=dtype)

    print(f"Compiling SYRK kernel to MLIR dialect: {args.dialect}")
    generate_mlir(model, inputs, args.out_mlir_path, args.dialect)


if __name__ == "__main__":
    main()
