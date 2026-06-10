#!/usr/bin/env python3
"""
PolyBench doitgen Kernel: Multi-Resolution Analysis

Reference: PolyBenchC-4.2.1/linear-algebra/kernels/doitgen/doitgen.c

Mathematical Operation:
    A[r, q, :] := A[r, q, :] * C4  (for all r, q)

Batched matrix-vector product across first two dimensions of 3D tensor.

This module implements the doitgen kernel as a PyTorch nn.Module and provides
MLIR generation capability via command-line interface.

Usage:
    # Direct execution for MLIR generation
    python kernel_doitgen.py ./output/doitgen_linalg.mlir --dialect linalg-on-tensors

    # Import as module
    from kernel_doitgen import Doitgen, init_array
    model = Doitgen(nr=150, nq=140, np=160)
    A, C4 = init_array(150, 140, 160)
    result = model(A, C4)
"""

import argparse

import torch
import torch.nn as nn

try:
    from PolyBenchPyTorch.linear_algebra.utils import (
        generate_mlir,
        make_parser,
        resolve_dtype,
    )
    from PolyBenchPyTorch.linear_algebra.utils import (
        get_dataset_dimensions as _get_dataset_dimensions,
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
    _get_dataset_dimensions = _mod.get_dataset_dimensions
    resolve_dtype = _mod.resolve_dtype
    generate_mlir = _mod.generate_mlir


def get_dataset_dimensions(dataset: str) -> dict:
    return _get_dataset_dimensions("doitgen", dataset)


class Doitgen(nn.Module):
    """
    PolyBench doitgen kernel: A[r,q,:] := A[r,q,:] * C4

    Multi-resolution analysis with 3D tensor transformation.

    Reference:
        PolyBenchC-4.2.1/linear-algebra/kernels/doitgen/doitgen.c

    Args:
        nr: First dimension of A (default: 150)
        nq: Second dimension of A (default: 140)
        np: Third dimension of A, both dimensions of C4 (default: 160)
    """

    def __init__(self, nr: int, nq: int, np: int) -> None:
        super().__init__()
        self.nr = nr
        self.nq = nq
        self.np = np

    def forward(
        self,
        A: torch.Tensor,
        C4: torch.Tensor,
    ) -> torch.Tensor:
        """
        Execute doitgen kernel computation.

        Adaptation from C: Nested loops converted to PyTorch tensor operations.
        C version computes sum[p] = sum_s(A[r][q][s] * C4[s][p]) for each (r, q),
        then A[r][q][p] = sum[p].

        Args:
            A: Input 3D tensor (nr, nq, np)
            C4: Transformation matrix (np, np)

        Returns:
            A_out: Transformed tensor (nr, nq, np) = A @ C4 (batched)

        Raises:
            RuntimeError: If tensor shapes are incompatible
        """
        # Shape assertions
        if not torch.jit.is_tracing():
            assert A.shape == (self.nr, self.nq, self.np), (
                f"A shape mismatch: {A.shape} != ({self.nr}, {self.nq}, {self.np})"
            )
            assert C4.shape == (self.np, self.np), (
                f"C4 shape mismatch: {C4.shape} != ({self.np}, {self.np})"
            )

        # Computation: A[r,q,:] := A[r,q,:] * C4 for all r, q
        # Using einsum for clarity: batch matrix-matrix multiply
        A_out = torch.einsum("rqs,sp->rqp", A, C4)

        return A_out


def init_array(
    nr: int, nq: int, np: int, dtype: torch.dtype = torch.float32
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Initialize arrays for doitgen kernel matching C reference implementation.

    Formulas from PolyBenchC-4.2.1/linear-algebra/kernels/doitgen/doitgen.c lines 44-52:
        A[i, j, k] = ((i*j+k) % np) / np
        C4[i, j] = (i*j % np) / np

    Args:
        nr: Dimension parameter
        nq: Dimension parameter
        np: Dimension parameter
        dtype: Tensor data type (default: torch.float32)

    Returns:
        Tuple of (A, C4):
            A (Tensor): (nr, nq, np) initialized 3D tensor
            C4 (Tensor): (np, np) initialized matrix
    """
    # Initialize A (nr x nq x np)
    A = torch.zeros((nr, nq, np), dtype=dtype)
    for i in range(nr):
        for j in range(nq):
            for k in range(np):
                A[i, j, k] = ((i * j + k) % np) / np

    # Initialize C4 (np x np)
    C4 = torch.zeros((np, np), dtype=dtype)
    for i in range(np):
        for j in range(np):
            C4[i, j] = (i * j % np) / np

    return A, C4


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for MLIR generation."""
    parser = make_parser("doitgen", "./output/doitgen_linalg.mlir")
    parser.add_argument("--nr", type=int, help="Dimension nr (overrides dataset value)")
    parser.add_argument("--nq", type=int, help="Dimension nq (overrides dataset value)")
    parser.add_argument("--np", type=int, help="Dimension np (overrides dataset value)")
    return parser.parse_args()


def main() -> None:
    """Generate MLIR from doitgen kernel model."""
    args = parse_args()

    dims = get_dataset_dimensions(args.dataset)
    nr = args.nr if args.nr is not None else dims["nr"]
    nq = args.nq if args.nq is not None else dims["nq"]
    np = args.np if args.np is not None else dims["np"]

    dtype = resolve_dtype(args.dtype)

    model = Doitgen(nr, nq, np)
    A, C4 = init_array(nr, nq, np, dtype=dtype)

    print(f"Compiling doitgen kernel to MLIR dialect: {args.dialect}")
    generate_mlir(model, (A, C4), args.out_mlir_path, args.dialect)


if __name__ == "__main__":
    main()
