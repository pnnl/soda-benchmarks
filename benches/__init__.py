"""Benchmark suites for soda-benchmarks.

This package is the single import identity for every benchmark kernel:

    from benches.PolyBenchPyTorch.linear_algebra.blas.gemm.gemm import Gemm

Importing kernels under any other name (e.g. a bare `PolyBenchPyTorch...`, which
only resolves when the interpreter starts inside `benches/`) creates a second,
unrelated copy of the same modules. Always spell it `benches.…`.

`sb_cli` imports this package to locate the tree, so nothing needs to guess paths
or depend on the current working directory. 
"""

from __future__ import annotations

from pathlib import Path

# The benches/ directory itself; holds the suites and experiments/.
ROOT = Path(__file__).resolve().parent

# The repository checkout containing benches/, scripts/, and sb_cli/.
REPO_ROOT = ROOT.parent

__all__ = ["REPO_ROOT", "ROOT"]
