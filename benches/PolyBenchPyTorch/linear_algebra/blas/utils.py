"""Backwards-compatibility re-export of shared linear algebra utilities.

The canonical implementations now live in
PolyBenchPyTorch.linear_algebra.utils. This module re-exports them so that
existing code importing from PolyBenchPyTorch.linear_algebra.blas.utils
continues to work without modification.
"""

from PolyBenchPyTorch.linear_algebra.utils import (  # noqa: F401
    gemm_init_array,
    get_dataset_dimensions,
)

__all__ = ["get_dataset_dimensions", "gemm_init_array"]
