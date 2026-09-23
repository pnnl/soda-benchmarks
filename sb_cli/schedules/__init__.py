"""Transform-schedule library for sb-cli.

A *schedule library* supplies the `transform.mlir` an experiment is scaffolded
with, for kernels whose schedule was produced somewhere other than sb-cli
itself -- typically by a design-space exploration run:

  schedules/<library>/<DATASET>/
    <kernel>.mlir                 # the schedule on its own
    <kernel>+<recipe>.mlir        # the same schedule, instrumented

`sb-cli init --default_ts <library>` copies the file matching the experiment's
kernel, dataset and `--instrumentation` recipe. The sentinel library
``"simple"`` means "whatever sb-cli would have written anyway" -- the recipe's
own `transform.mlir`, or the no-op template -- and is always available.

Only the schedule comes from here. The IP files an instrumented schedule needs
still come from the recipe (see `sb_cli.recipes`), because the merged files call
exactly the same instrumentation passes.

Why instrumentation is merged in rather than appended
-----------------------------------------------------
The SODA instrumentation passes walk `scf::ForOp` only (see
`examples/soda-plugins/lib/sodap/InstrPasses.cpp`). A DSE schedule tiles with
`transform.structured.tile_using_for`, which produces `scf.for`, and then lowers
what is left with `convert-linalg-to-affine-loops`, after which every loop is an
`affine.for`. So the instrumentation pass has to run *between* those two steps.
Appending it to the end of a schedule parses and runs cleanly but instruments
nothing at all, which is why the merged files are generated and checked in
rather than assembled at scaffold time.

New libraries are added by dropping a folder here with the layout above; no code
changes are required.
"""

from __future__ import annotations

from pathlib import Path

from sb_cli.recipes import NONE_RECIPE

# Sentinel library name meaning "use the schedule sb-cli generates itself".
SIMPLE_SCHEDULE = "simple"

_SCHEDULES_DIR = Path(__file__).parent


class ScheduleUnavailableError(ValueError):
    """Raised when a library has no schedule for a kernel/dataset/recipe.

    A subclass of ValueError so callers that already handle the unknown-recipe
    case keep working unchanged.
    """


def _library_dir(library: str) -> Path:
    return _SCHEDULES_DIR / library


def available_schedules() -> list[str]:
    """Return all selectable library names, including the ``simple`` sentinel.

    A directory qualifies as a library if it holds at least one
    ``<DATASET>/<kernel>.mlir``.
    """
    names = [SIMPLE_SCHEDULE]
    if _SCHEDULES_DIR.is_dir():
        for child in sorted(_SCHEDULES_DIR.iterdir()):
            if child.is_dir() and any(child.glob("*/*.mlir")):
                names.append(child.name)
    return names


def available_datasets(library: str) -> list[str]:
    """Return the datasets `library` carries schedules for."""
    lib_dir = _library_dir(library)
    if not lib_dir.is_dir():
        return []
    return sorted(
        d.name for d in lib_dir.iterdir() if d.is_dir() and any(d.glob("*.mlir"))
    )


def available_kernels(library: str, dataset: str) -> list[str]:
    """Return the kernels `library` carries a bare schedule for under `dataset`."""
    ds_dir = _library_dir(library) / dataset
    if not ds_dir.is_dir():
        return []
    return sorted(p.stem for p in ds_dir.glob("*.mlir") if "+" not in p.stem)


def available_instrumentation(library: str, dataset: str, kernel: str) -> list[str]:
    """Return the recipes `library` has a merged schedule for, plus ``none``."""
    ds_dir = _library_dir(library) / dataset
    if not (ds_dir / f"{kernel}.mlir").is_file():
        return []
    merged = sorted(p.stem.split("+", 1)[1] for p in ds_dir.glob(f"{kernel}+*.mlir"))
    return [NONE_RECIPE, *merged]


def resolve_schedule(
    library: str,
    *,
    benchmark: str | None,
    dataset: str,
    instrumentation: str,
) -> Path | None:
    """Resolve a library selection to the ``.mlir`` to copy as transform.mlir.

    Args:
        library: Library name, or ``"simple"`` for the sentinel.
        benchmark: Kernel short name (e.g. ``gemm``), or None when the
            experiment has no benchmark.
        dataset: Dataset name, e.g. ``MINI``.
        instrumentation: Recipe name, or ``"none"``.

    Returns:
        The schedule to copy, or ``None`` for the ``simple`` sentinel -- meaning
        the caller should fall back to the recipe's schedule or the template.

    Raises:
        ScheduleUnavailableError: If `library` is unknown, or has nothing for
            this kernel/dataset/recipe combination. The message lists what is
            available so the caller can correct the request.
    """
    if library == SIMPLE_SCHEDULE:
        return None

    lib_dir = _library_dir(library)
    if not lib_dir.is_dir():
        raise ScheduleUnavailableError(
            f"Unknown transform schedule: '{library}'. "
            f"Choose from {available_schedules()}"
        )

    if benchmark is None:
        raise ScheduleUnavailableError(
            f"Transform schedule '{library}' needs --benchmark_name: there is no "
            "kernel to look a schedule up for."
        )

    datasets = available_datasets(library)
    if dataset not in datasets:
        raise ScheduleUnavailableError(
            f"Transform schedule '{library}' has no schedules for dataset "
            f"'{dataset}'. Available datasets: {', '.join(datasets)}"
        )

    kernels = available_kernels(library, dataset)
    if benchmark not in kernels:
        raise ScheduleUnavailableError(
            f"Transform schedule '{library}' has no schedule for kernel "
            f"'{benchmark}' with dataset '{dataset}'. "
            f"Available kernels: {', '.join(kernels)}"
        )

    stem = (
        benchmark
        if instrumentation == NONE_RECIPE
        else f"{benchmark}+{instrumentation}"
    )
    path = lib_dir / dataset / f"{stem}.mlir"
    if not path.is_file():
        variants = available_instrumentation(library, dataset, benchmark)
        raise ScheduleUnavailableError(
            f"Transform schedule '{library}' has no '{instrumentation}' "
            f"instrumentation variant for kernel '{benchmark}' with dataset "
            f"'{dataset}'. Available: {', '.join(variants)}"
        )
    return path
