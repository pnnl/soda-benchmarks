#!/usr/bin/env python3
"""PolyBench GESUMMV Kernel: y := alpha*(A*x) + beta*(B*x)

Reference: PolyBenchC-4.2.1/linear-algebra/blas/gesummv/gesummv.c
"""

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


class Gesummv(nn.Module):
    def __init__(self, n: int) -> None:
        super().__init__()
        self.n = n

    def forward(
        self,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if not torch.jit.is_tracing():
            assert A.shape == (self.n, self.n)
            assert B.shape == (self.n, self.n)
            assert x.shape == (self.n,)

        y = alpha * torch.mv(A, x) + beta * torch.mv(B, x)
        return y


def init_array(
    n: int, dtype: torch.dtype = torch.float32
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    alpha = torch.tensor(1.5, dtype=dtype)
    beta = torch.tensor(1.2, dtype=dtype)

    A = torch.zeros((n, n), dtype=dtype)
    B = torch.zeros((n, n), dtype=dtype)
    x = torch.zeros(n, dtype=dtype)

    for i in range(n):
        x[i] = (i % n) / float(n)
        for j in range(n):
            A[i, j] = ((i * j + 1) % n) / float(n)
            B[i, j] = ((i * j + 2) % n) / float(n)

    return alpha, beta, A, B, x


def parse_args() -> argparse.Namespace:
    parser = make_parser("gesummv", "./output/gesummv_linalg.mlir")
    parser.add_argument("--n", type=int, help="Dimension n (overrides dataset)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dims = get_dataset_dimensions("gesummv", args.dataset)
    n = args.n if args.n is not None else dims["n"]

    dtype = resolve_dtype(args.dtype)

    model = Gesummv(n)
    alpha, beta, A, B, x = init_array(n, dtype=dtype)

    print(f"Compiling GESUMMV kernel to MLIR dialect: {args.dialect}")
    generate_mlir(model, (alpha, beta, A, B, x), args.out_mlir_path, args.dialect)


if __name__ == "__main__":
    main()
