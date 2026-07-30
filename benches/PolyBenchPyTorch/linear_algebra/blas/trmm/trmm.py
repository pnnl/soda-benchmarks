#!/usr/bin/env python3
"""PolyBench TRMM Kernel: B := alpha*A^T*B where A is lower triangular."""

import argparse

import torch
import torch.nn as nn

from benches.PolyBenchPyTorch.linear_algebra.utils import (
    generate_mlir,
    get_dataset_dimensions,
    make_parser,
    resolve_dtype,
)


class Trmm(nn.Module):
    def __init__(self, m: int, n: int) -> None:
        super().__init__()
        self.m = m
        self.n = n
        # Precomputed buffers avoid torch.tril/torch.eye in forward() (not TOSA-legal)
        self.register_buffer("lower_mask", torch.tril(torch.ones(m, m)))
        self.register_buffer("eye", torch.eye(m))

    def forward(
        self, alpha: torch.Tensor, A: torch.Tensor, B: torch.Tensor
    ) -> torch.Tensor:
        if not torch.jit.is_tracing():
            assert A.shape == (self.m, self.m)
            assert B.shape == (self.m, self.n)

        # Extract lower triangle and set unit diagonal using precomputed buffers
        A_lower = A * self.lower_mask + self.eye
        B_out = alpha * torch.matmul(A_lower.t(), B)
        return B_out


def init_array(m: int, n: int, dtype: torch.dtype = torch.float32):
    alpha = torch.tensor(1.5, dtype=dtype)
    A = torch.zeros((m, m), dtype=dtype)
    B = torch.zeros((m, n), dtype=dtype)

    for i in range(m):
        for j in range(i):
            A[i, j] = ((i + j) % m) / float(m)
        A[i, i] = 1.0
        for j in range(n):
            B[i, j] = ((n + (i - j)) % n) / float(n)

    return alpha, A, B


def parse_args() -> argparse.Namespace:
    parser = make_parser("trmm", "./output/trmm_linalg.mlir")
    parser.add_argument("--m", type=int, help="Dimension m (overrides dataset)")
    parser.add_argument("--n", type=int, help="Dimension n (overrides dataset)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dims = get_dataset_dimensions("trmm", args.dataset)
    m = args.m if args.m is not None else dims["m"]
    n = args.n if args.n is not None else dims["n"]

    dtype = resolve_dtype(args.dtype)

    model = Trmm(m, n)
    inputs = init_array(m, n, dtype=dtype)

    print(f"Compiling TRMM kernel to MLIR dialect: {args.dialect}")
    generate_mlir(model, inputs, args.out_mlir_path, args.dialect)


if __name__ == "__main__":
    main()
