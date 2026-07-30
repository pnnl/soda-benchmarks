#!/usr/bin/env python3
"""PolyBench GEMVER Kernel

Reference: PolyBenchC-4.2.1/linear-algebra/blas/gemver/gemver.c

Operation: Multi-step updates involving A, u1, v1, u2, v2, x, y, z, w
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


class Gemver(nn.Module):
    """GEMVER kernel (matrix-vector updates)

    Args:
        n: Size of square matrix A and vectors
    """

    def __init__(self, n: int) -> None:
        super().__init__()
        self.n = n

    def forward(
        self,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        A: torch.Tensor,
        u1: torch.Tensor,
        v1: torch.Tensor,
        u2: torch.Tensor,
        v2: torch.Tensor,
        w: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        # Shape checks
        if not torch.jit.is_tracing():
            assert A.shape == (self.n, self.n)
            assert u1.shape == (self.n,)
            assert v1.shape == (self.n,)
            assert u2.shape == (self.n,)
            assert v2.shape == (self.n,)
            assert w.shape == (self.n,)
            assert x.shape == (self.n,)
            assert y.shape == (self.n,)
            assert z.shape == (self.n,)

        # Step 1: A := A + u1*v1^T + u2*v2^T
        A = A + u1.unsqueeze(1) * v1.unsqueeze(0) + u2.unsqueeze(1) * v2.unsqueeze(0)

        # Step 2: x := beta * A^T * y + x
        x = x + beta * torch.mv(A.t(), y)

        # Step 3: x := x + z
        x = x + z

        # Step 4: w := w + alpha * A * x
        w = w + alpha * torch.mv(A, x)

        return w


def init_array(n: int, dtype: torch.dtype = torch.float32):
    """Initialize arrays for GEMVER according to data-model.md

    Returns (alpha, beta, A, u1, v1, u2, v2, w, x, y, z).
    """
    alpha = torch.tensor(1.5, dtype=dtype)
    beta = torch.tensor(1.2, dtype=dtype)

    A = torch.zeros((n, n), dtype=dtype)
    u1 = torch.zeros(n, dtype=dtype)
    u2 = torch.zeros(n, dtype=dtype)
    v1 = torch.zeros(n, dtype=dtype)
    v2 = torch.zeros(n, dtype=dtype)
    y = torch.zeros(n, dtype=dtype)
    z = torch.zeros(n, dtype=dtype)
    x = torch.zeros(n, dtype=dtype)
    w = torch.zeros(n, dtype=dtype)

    fn = float(n)
    for i in range(n):
        for j in range(n):
            A[i, j] = ((i * j) % n) / float(n)

        u1[i] = i
        u2[i] = ((i + 1) / fn) / 2.0
        v1[i] = ((i + 1) / fn) / 4.0
        v2[i] = ((i + 1) / fn) / 6.0
        y[i] = ((i + 1) / fn) / 8.0
        z[i] = ((i + 1) / fn) / 9.0
        x[i] = 0.0
        w[i] = 0.0

    return alpha, beta, A, u1, v1, u2, v2, w, x, y, z


def parse_args() -> argparse.Namespace:
    parser = make_parser("gemver", "./output/gemver_linalg.mlir")
    parser.add_argument("--n", type=int, help="Dimension n (overrides dataset value)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dims = get_dataset_dimensions("gemver", args.dataset)
    n = args.n if args.n is not None else dims["n"]

    dtype = resolve_dtype(args.dtype)

    model = Gemver(n)
    inputs = init_array(n, dtype=dtype)

    print(f"Compiling GEMVER kernel to MLIR dialect: {args.dialect}")
    generate_mlir(model, inputs, args.out_mlir_path, args.dialect)


if __name__ == "__main__":
    main()
