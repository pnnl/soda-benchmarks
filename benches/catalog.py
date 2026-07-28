"""Catalog of the benchmark kernels available under `benches/`.

Tools such as `sb-cli` ask this module which kernels exist and where they live,
instead of deriving module paths by string manipulation. That keeps exactly one
place aware of the layout convention:

    <suite>/.../<kernel>/<kernel>.py

defining an `nn.Module` subclass and an `init_array` function, with shared
helpers in a `utils.py` somewhere above it.

Discovery is filesystem-only, so listing benchmarks never imports torch. Only
`model_class` imports a kernel, and only the one that was asked for.
"""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from benches import ROOT

# Directories under benches/ that never contain kernels
_SKIP_DIRS = frozenset({"experiments", "tests", "__pycache__", "PolyBenchC-4.2.1"})

# Names a utils module must provide to be usable by a generated torchscript.py.
# `benches/PolyBenchPyTorch/linear_algebra/blas/utils.py` is a partial
# back-compat shim, so proximity alone is not enough to pick the right one.
_REQUIRED_UTILS = ("generate_mlir", "get_dataset_dimensions", "resolve_dtype")

# Fallback when a suite ships no qualifying utils module
_DEFAULT_UTILS = "benches.PolyBenchPyTorch.linear_algebra.utils"

_ROOT_PACKAGE = ROOT.name


@dataclass(frozen=True)
class Benchmark:
    """A single benchmark kernel.

    Attributes:
        name: Short kernel name, e.g. `gemm`.
        package: Dotted path of the kernel package,
            e.g. `benches.PolyBenchPyTorch.linear_algebra.blas.gemm`.
        module: Dotted path of the implementation module holding the
            `nn.Module` subclass and `init_array`, e.g. `<package>.gemm`.
        utils_module: Dotted path of the shared helpers module providing
            `generate_mlir`, `get_dataset_dimensions`, and `resolve_dtype`.
    """

    name: str
    package: str
    module: str
    utils_module: str


def _dotted(path: Path) -> str:
    """Return the dotted module path for a file or directory under benches/."""
    rel = path.relative_to(ROOT)
    parts = [*rel.parts[:-1], rel.stem] if path.suffix == ".py" else list(rel.parts)
    return ".".join([_ROOT_PACKAGE, *parts])


def _provides_required_helpers(utils_path: Path) -> bool:
    """Return True if `utils_path` defines or re-exports every required helper.

    Parsed with `ast` rather than imported, so discovery stays torch-free.
    """
    import ast

    try:
        tree = ast.parse(utils_path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return False

    exported: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            exported.add(node.name)
        elif isinstance(node, ast.ImportFrom):
            exported.update(alias.asname or alias.name for alias in node.names)
    return all(name in exported for name in _REQUIRED_UTILS)


def _find_utils_module(kernel_dir: Path) -> str:
    """Return the dotted path of the utils module serving `kernel_dir`.

    Walks upward from the kernel package to benches/, taking the nearest
    `utils.py` that provides all of `_REQUIRED_UTILS`.
    """
    for parent in kernel_dir.parents:
        candidate = parent / "utils.py"
        if candidate.is_file() and _provides_required_helpers(candidate):
            return _dotted(candidate)
        if parent == ROOT:
            break
    return _DEFAULT_UTILS


def _walk_kernel_dirs(directory: Path) -> list[Path]:
    """Return kernel package directories (those containing `<name>/<name>.py`)."""
    found: list[Path] = []
    for child in sorted(directory.iterdir()):
        if not child.is_dir() or child.name in _SKIP_DIRS:
            continue
        if not child.name.isidentifier():
            continue
        if (child / f"{child.name}.py").is_file():
            found.append(child)
            continue
        found.extend(_walk_kernel_dirs(child))
    return found


@lru_cache(maxsize=1)
def available() -> dict[str, Benchmark]:
    """Return every discovered benchmark, keyed by short name.

    Returns:
        Mapping of kernel name to `Benchmark`, sorted by name.
    """
    benchmarks: dict[str, Benchmark] = {}
    for kernel_dir in _walk_kernel_dirs(ROOT):
        package = _dotted(kernel_dir)
        benchmarks[kernel_dir.name] = Benchmark(
            name=kernel_dir.name,
            package=package,
            module=f"{package}.{kernel_dir.name}",
            utils_module=_find_utils_module(kernel_dir),
        )
    return dict(sorted(benchmarks.items()))


def find(name_or_path: str) -> Benchmark | None:
    """Look up a benchmark by any of its accepted spellings.

    Accepts the short name (`gemm`), a dotted package path with or without the
    `benches.` prefix (`PolyBenchPyTorch.linear_algebra.blas.gemm`), and paths
    that already name the implementation module (`...blas.gemm.gemm`).

    Args:
        name_or_path: Benchmark name or dotted module path.

    Returns:
        The matching `Benchmark`, or None if nothing matched.
    """
    benchmarks = available()
    if name_or_path in benchmarks:
        return benchmarks[name_or_path]

    query = name_or_path.strip(".")
    if not query.startswith(f"{_ROOT_PACKAGE}."):
        query = f"{_ROOT_PACKAGE}.{query}"

    for bench in benchmarks.values():
        if query in (bench.package, bench.module):
            return bench
    return None


def model_class(bench: Benchmark) -> str:
    """Return the name of the kernel's `nn.Module` subclass.

    Imports `bench.module`; kernel class names do not follow a mechanical rule
    (`gemm` -> `Gemm`, but `twomm` -> `TwoMM`), so the class is located by
    inspection rather than guessed from the name.

    Args:
        bench: The benchmark to inspect.

    Returns:
        Class name, falling back to the capitalized kernel name if no
        `nn.Module` subclass is found.

    Raises:
        ImportError: If the implementation module cannot be imported.
    """
    import torch.nn as nn

    module = importlib.import_module(bench.module)
    for attr_name, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, nn.Module) and obj is not nn.Module:
            return attr_name
    return bench.name.capitalize()
