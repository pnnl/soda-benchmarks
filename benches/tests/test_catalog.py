"""Tests for benches.catalog — benchmark discovery and name resolution.

Also guards the packaging rules this repo depends on: `benches` is importable
because it is installed (editable), not because of a PYTHONPATH shortcut, and
every kernel must be importable as `benches.<suite>....` from any directory.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version

import pytest

import benches
from benches import catalog


class TestAvailable:
    def test_discovers_known_kernels(self) -> None:
        """Discovery finds the PolyBench kernels shipped in the repo."""
        names = catalog.available()
        assert {"gemm", "twomm", "threemm", "trmm"} <= set(names)

    def test_modules_are_fully_qualified(self) -> None:
        """Every discovered path is rooted at the `benches` package."""
        for bench in catalog.available().values():
            assert bench.package.startswith("benches.")
            assert bench.module == f"{bench.package}.{bench.name}"
            assert bench.utils_module.startswith("benches.")

    def test_skips_non_kernel_directories(self) -> None:
        """experiments/ and tests/ are not mistaken for kernels."""
        names = catalog.available()
        assert "experiments" not in names
        assert "tests" not in names


class TestFind:
    def test_spellings_resolve_to_same_benchmark(self) -> None:
        """Short name, dotted path, and qualified path all resolve equally."""
        expected = catalog.available()["gemm"]
        for spelling in [
            "gemm",
            "PolyBenchPyTorch.linear_algebra.blas.gemm",
            "benches.PolyBenchPyTorch.linear_algebra.blas.gemm",
            "benches.PolyBenchPyTorch.linear_algebra.blas.gemm.gemm",
        ]:
            assert catalog.find(spelling) == expected, spelling

    def test_unknown_returns_none(self) -> None:
        """An unrecognized name resolves to None rather than raising."""
        assert catalog.find("no.such.module.Xyz") is None
        assert catalog.find("nope") is None


class TestPackaging:
    def test_repo_is_installed_as_a_distribution(self) -> None:
        """`benches` comes from an install, not a sys.path/PYTHONPATH shortcut.

        Fails if someone drops the editable install and reinstates a path hack;
        run `pixi install` to fix.
        """
        try:
            assert version("soda-benchmarks")
        except PackageNotFoundError:  # pragma: no cover - only without install
            pytest.fail(
                "soda-benchmarks is not installed; run `pixi install` "
                "(the repo must be installed editable, not put on PYTHONPATH)"
            )

    def test_install_points_at_the_source_tree(self) -> None:
        """The install is editable: imports resolve to the working copy."""
        assert benches.ROOT.is_dir()
        assert (benches.REPO_ROOT / "pyproject.toml").is_file()


class TestImportIdentity:
    @pytest.mark.slow
    def test_every_kernel_imports_under_canonical_name(self) -> None:
        """Each kernel module imports as `benches....` (needs torch)."""
        for bench in catalog.available().values():
            importlib.import_module(bench.module)

    def test_utils_modules_import(self) -> None:
        """Shared utils modules import under the qualified name.

        `linear_algebra/blas/utils.py` re-exports from its parent; before the
        single-identity fix it raised ModuleNotFoundError.
        """
        for module in {b.utils_module for b in catalog.available().values()}:
            importlib.import_module(module)
        importlib.import_module("benches.PolyBenchPyTorch.linear_algebra.blas.utils")

    def test_kernel_imports_from_unrelated_cwd(self, tmp_path) -> None:
        """A kernel imports from a directory outside the repo."""
        bench = catalog.available()["gemm"]
        result = subprocess.run(
            [sys.executable, "-c", f"import {bench.module}"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
