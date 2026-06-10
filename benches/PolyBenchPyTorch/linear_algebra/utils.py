"""Shared utilities for PolyBench PyTorch linear algebra kernels.

Provides common argument parsing, dataset dimension lookup, dtype resolution,
and MLIR generation helpers used by all kernels under
PolyBenchPyTorch/linear_algebra/ (both kernels/ and blas/ subtrees).

Reference: PolyBenchC-4.2.1
"""

import argparse
import os
import sys

import torch
import torch.nn as nn

__all__ = [
    "make_parser",
    "get_dataset_dimensions",
    "resolve_dtype",
    "generate_mlir",
    "gemm_init_array",
]

# ---------------------------------------------------------------------------
# Dataset dimension registry — single authoritative source for all kernels
# ---------------------------------------------------------------------------

_DATASETS: dict[str, dict[str, dict[str, int]]] = {
    "2mm": {
        "MINI": {"ni": 16, "nj": 18, "nk": 22, "nl": 24},
        "SMALL": {"ni": 40, "nj": 50, "nk": 70, "nl": 80},
        "MEDIUM": {"ni": 180, "nj": 190, "nk": 210, "nl": 220},
        "LARGE": {"ni": 800, "nj": 900, "nk": 1100, "nl": 1200},
        "EXTRALARGE": {"ni": 1600, "nj": 1800, "nk": 2200, "nl": 2400},
    },
    "3mm": {
        "MINI": {"ni": 16, "nj": 18, "nk": 20, "nl": 22, "nm": 24},
        "SMALL": {"ni": 40, "nj": 50, "nk": 60, "nl": 70, "nm": 80},
        "MEDIUM": {"ni": 180, "nj": 190, "nk": 200, "nl": 210, "nm": 220},
        "LARGE": {"ni": 800, "nj": 900, "nk": 1000, "nl": 1100, "nm": 1200},
        "EXTRALARGE": {"ni": 1600, "nj": 1800, "nk": 2000, "nl": 2200, "nm": 2400},
    },
    "atax": {
        "MINI": {"m": 38, "n": 42},
        "SMALL": {"m": 116, "n": 124},
        "MEDIUM": {"m": 390, "n": 410},
        "LARGE": {"m": 1900, "n": 2100},
        "EXTRALARGE": {"m": 1800, "n": 2200},
    },
    "bicg": {
        "MINI": {"m": 38, "n": 42},
        "SMALL": {"m": 116, "n": 124},
        "MEDIUM": {"m": 390, "n": 410},
        "LARGE": {"m": 1900, "n": 2100},
        "EXTRALARGE": {"m": 1800, "n": 2200},
    },
    "doitgen": {
        "MINI": {"nr": 10, "nq": 8, "np": 12},
        "SMALL": {"nr": 25, "nq": 20, "np": 30},
        "MEDIUM": {"nr": 50, "nq": 40, "np": 60},
        "LARGE": {"nr": 150, "nq": 140, "np": 160},
        "EXTRALARGE": {"nr": 250, "nq": 220, "np": 270},
    },
    "mvt": {
        "MINI": {"n": 40},
        "SMALL": {"n": 120},
        "MEDIUM": {"n": 400},
        "LARGE": {"n": 2000},
        "EXTRALARGE": {"n": 4000},
    },
    "gemm": {
        "MINI": {"ni": 20, "nj": 25, "nk": 30},
        "SMALL": {"ni": 60, "nj": 70, "nk": 80},
        "MEDIUM": {"ni": 200, "nj": 220, "nk": 240},
        "LARGE": {"ni": 1000, "nj": 1100, "nk": 1200},
        "EXTRALARGE": {"ni": 2000, "nj": 2300, "nk": 2600},
    },
    "gemver": {
        "MINI": {"n": 40},
        "SMALL": {"n": 120},
        "MEDIUM": {"n": 400},
        "LARGE": {"n": 2000},
        "EXTRALARGE": {"n": 4000},
    },
    "gesummv": {
        "MINI": {"n": 30},
        "SMALL": {"n": 90},
        "MEDIUM": {"n": 250},
        "LARGE": {"n": 1300},
        "EXTRALARGE": {"n": 2800},
    },
    "symm": {
        "MINI": {"m": 20, "n": 30},
        "SMALL": {"m": 60, "n": 80},
        "MEDIUM": {"m": 200, "n": 240},
        "LARGE": {"m": 1000, "n": 1200},
        "EXTRALARGE": {"m": 2000, "n": 2600},
    },
    "syr2k": {
        "MINI": {"m": 20, "n": 30},
        "SMALL": {"m": 60, "n": 80},
        "MEDIUM": {"m": 200, "n": 240},
        "LARGE": {"m": 1000, "n": 1200},
        "EXTRALARGE": {"m": 2000, "n": 2600},
    },
    "syrk": {
        "MINI": {"m": 20, "n": 30},
        "SMALL": {"m": 60, "n": 80},
        "MEDIUM": {"m": 200, "n": 240},
        "LARGE": {"m": 1000, "n": 1200},
        "EXTRALARGE": {"m": 2000, "n": 2600},
    },
    "trmm": {
        "MINI": {"m": 20, "n": 30},
        "SMALL": {"m": 60, "n": 80},
        "MEDIUM": {"m": 200, "n": 240},
        "LARGE": {"m": 1000, "n": 1200},
        "EXTRALARGE": {"m": 2000, "n": 2600},
    },
}

_DTYPE_MAP: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
}

_DIALECT_CHOICES = ["linalg-on-tensors", "tosa", "torch", "raw", "mhlo"]
_DATASET_CHOICES = ["MINI", "SMALL", "MEDIUM", "LARGE", "EXTRALARGE"]
_BASE_DTYPE_CHOICES = ["float16", "float32", "float64"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def make_parser(
    kernel_name: str,
    default_out_path: str,
    extra_dtype_choices: list[str] | None = None,
) -> argparse.ArgumentParser:
    """Return an ArgumentParser pre-loaded with common PolyBench flags.

    Adds: out_mlir_path (positional), --dialect, --dataset, --dtype.
    The caller should add kernel-specific dimension flags (--ni, --m, etc.)
    before calling parser.parse_args().

    Args:
        kernel_name: Human-readable kernel name used in the description string.
        default_out_path: Default value for the out_mlir_path positional argument.
        extra_dtype_choices: Additional dtype strings beyond float16/float32/float64
            (e.g. ["int32", "int16", "int64"] for the 3mm kernel).

    Returns:
        Configured ArgumentParser instance.
    """
    dtype_choices = _BASE_DTYPE_CHOICES + (extra_dtype_choices or [])

    parser = argparse.ArgumentParser(
        description=f"Generate MLIR for PolyBench {kernel_name} kernel"
    )
    parser.add_argument(
        "out_mlir_path",
        nargs="?",
        default=default_out_path,
        help=f"Output MLIR file path (default: {default_out_path})",
    )
    parser.add_argument(
        "--dialect",
        type=str,
        default="tosa",
        choices=_DIALECT_CHOICES,
        help="MLIR dialect (default: tosa)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="MINI",
        choices=_DATASET_CHOICES,
        help="Dataset size (default: MINI)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=dtype_choices,
        help="Tensor data type (default: float32)",
    )
    return parser


def get_dataset_dimensions(kernel: str, dataset: str) -> dict[str, int]:
    """Return dimension dict for the given kernel and dataset size.

    Args:
        kernel: Lowercase kernel name (e.g. 'gemm', '2mm', 'atax').
        dataset: Dataset size string — one of MINI, SMALL, MEDIUM, LARGE, EXTRALARGE.

    Returns:
        Dictionary mapping dimension names to integer values.

    Raises:
        ValueError: If kernel or dataset is not recognised.
    """
    dataset = dataset.upper()
    if kernel not in _DATASETS:
        raise ValueError(
            f"Unknown kernel: '{kernel}'. Known kernels: {sorted(_DATASETS)}"
        )
    if dataset not in _DATASETS[kernel]:
        raise ValueError(
            f"Unknown dataset: '{dataset}'. Choose from {_DATASET_CHOICES}"
        )
    return _DATASETS[kernel][dataset]


def resolve_dtype(dtype_str: str) -> torch.dtype:
    """Map a dtype string to the corresponding torch.dtype.

    Args:
        dtype_str: One of float16, float32, float64, int16, int32, int64.

    Returns:
        Corresponding torch.dtype value.

    Raises:
        ValueError: If dtype_str is not recognised.
    """
    if dtype_str not in _DTYPE_MAP:
        raise ValueError(
            f"Unknown dtype: '{dtype_str}'. Choose from {list(_DTYPE_MAP)}"
        )
    return _DTYPE_MAP[dtype_str]


def generate_mlir(
    model: nn.Module,
    inputs: tuple[torch.Tensor, ...],
    out_path: str,
    dialect: str,
) -> None:
    """Compile model to MLIR and write the result to out_path.

    Imports torch_mlir inside the function so that the ImportError is scoped
    here and does not propagate at module import time.

    Args:
        model: PyTorch nn.Module to compile.
        inputs: Tuple of example input tensors (used for tracing).
        out_path: Output file path for the generated MLIR text.
        dialect: MLIR output dialect (e.g. 'tosa', 'linalg-on-tensors').

    Raises:
        SystemExit: If torch_mlir is not installed.
    """
    try:
        from torch_mlir import torchscript
    except ImportError:
        print("ERROR: torch-mlir is not installed.")
        print("MLIR generation requires the torch-mlir library.")
        print(
            "See specs/research.md for installation instructions "
            "(source build required on aarch64)."
        )
        sys.exit(1)

    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)

    mlir_module = torchscript.compile(
        model,
        inputs,
        output_type=dialect,
        use_tracing=True,
    )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(str(mlir_module))

    print(f"MLIR written to {out_path}")


def gemm_init_array(
    ni: int,
    nj: int,
    nk: int,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Initialize arrays for the GEMM kernel following PolyBenchC formulas.

    Moved from PolyBenchPyTorch/linear_algebra/blas/utils.py.
    Kept here for backwards compatibility with callers that import it from
    either location.

    Args:
        ni: Rows in A and C.
        nj: Columns in B and C.
        nk: Columns in A, rows in B.
        dtype: Tensor data type (default: torch.float32).

    Returns:
        Tuple (alpha, beta, A, B, C) where alpha and beta are 0-dimensional
        tensors as required by torch-mlir.
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
