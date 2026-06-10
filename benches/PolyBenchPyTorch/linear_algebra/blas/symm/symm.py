#!/usr/bin/env python3
"""PolyBench SYMM Kernel: C := alpha*A*B + beta*C where A is symmetric."""

import argparse

import torch
import torch.nn as nn

try:
    from PolyBenchPyTorch.linear_algebra.utils import (
        generate_mlir,
        get_dataset_dimensions,
        make_parser,
        resolve_dtype,
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
    get_dataset_dimensions = _mod.get_dataset_dimensions
    resolve_dtype = _mod.resolve_dtype
    generate_mlir = _mod.generate_mlir


class Symm(nn.Module):
    def __init__(self, m: int, n: int) -> None:
        super().__init__()
        self.m = m
        self.n = n
        # Precomputed mask avoids torch.tril in forward() (not TOSA-legal)
        self.register_buffer("mask", torch.tril(torch.ones(m, m)))

    def forward(
        self,
        alpha: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        beta: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        if not torch.jit.is_tracing():
            assert A.shape == (self.m, self.m)
            assert B.shape == (self.m, self.n)
            assert C.shape == (self.m, self.n)

        # Reconstruct symmetric A from lower triangle using precomputed mask
        A_sym = A * self.mask + A.t() * (1.0 - self.mask)
        C_out = alpha * torch.matmul(A_sym, B) + beta * C
        return C_out


def init_array(m: int, n: int, dtype: torch.dtype = torch.float32):
    alpha = torch.tensor(1.5, dtype=dtype)
    beta = torch.tensor(1.2, dtype=dtype)
    C = torch.zeros((m, n), dtype=dtype)
    B = torch.zeros((m, n), dtype=dtype)
    A = torch.zeros((m, m), dtype=dtype)

    for i in range(m):
        for j in range(n):
            C[i, j] = ((i + j) % 100) / float(m)
            B[i, j] = ((n + i - j) % 100) / float(m)
        for j in range(i + 1):
            A[i, j] = ((i + j) % 100) / float(m)
        for j in range(i + 1, m):
            A[i, j] = -999

    return alpha, beta, A, B, C


def parse_args() -> argparse.Namespace:
    parser = make_parser("symm", "./output/symm_linalg.mlir")
    parser.add_argument("--m", type=int, help="Dimension m (overrides dataset)")
    parser.add_argument("--n", type=int, help="Dimension n (overrides dataset)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dims = get_dataset_dimensions("symm", args.dataset)
    m = args.m if args.m is not None else dims["m"]
    n = args.n if args.n is not None else dims["n"]

    dtype = resolve_dtype(args.dtype)

    model = Symm(m, n)
    alpha, beta, A, B, C = init_array(m, n, dtype=dtype)

    print(f"Compiling SYMM kernel to MLIR dialect: {args.dialect}")
    generate_mlir(model, (alpha, A, B, beta, C), args.out_mlir_path, args.dialect)


if __name__ == "__main__":
    main()
