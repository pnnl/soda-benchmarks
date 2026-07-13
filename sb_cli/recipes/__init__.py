"""Instrumentation recipe library for sb-cli.

A *recipe* bundles everything needed to add one kind of hardware
instrumentation to a synthesis experiment:

  recipes/<name>/
    transform.mlir        # transform-dialect schedule invoking the SODA pass
    IPs/
      <module>.v          # Verilog IP integrated by Bambu
      <module>.c          # C emulation stub (excluded from HLS parsing)
      module_lib.h        # C declarations
      module_lib.xml      # Bambu interface library
      constraints_STD.xml # Bambu resource constraints

`sb-cli init --instrumentation <name>` copies the recipe's `transform.mlir`
and `IPs/` into the generated experiment and wires the Bambu IP-integration
Makefile variables so the IP is only synthesized when the schedule references
it. The sentinel recipe ``"none"`` performs no instrumentation (no-op schedule,
no IP integration) and is always available.

New recipes are added simply by dropping a folder here with the layout above;
no code changes are required.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Sentinel recipe name meaning "no instrumentation".
NONE_RECIPE = "none"

_RECIPES_DIR = Path(__file__).parent


@dataclass(frozen=True)
class Recipe:
    """Resolved paths for a single instrumentation recipe.

    Attributes:
        name: Recipe name (folder name under recipes/).
        transform_mlir: Path to the recipe's transform.mlir schedule.
        ip_dir: Path to the recipe's IPs/ directory.
        verilog_inputs: Verilog IP source files (``*.v``).
        c_excludes: C emulation stubs excluded from HLS parsing (``*.c``).
        module_lib: Bambu interface library (``module_lib.xml``).
        constraints: Bambu resource constraints (``constraints_STD.xml``).
    """

    name: str
    transform_mlir: Path
    ip_dir: Path
    verilog_inputs: list[Path]
    c_excludes: list[Path]
    module_lib: Path | None
    constraints: Path | None


def available_recipes() -> list[str]:
    """Return all selectable recipe names, including the ``none`` sentinel.

    A directory qualifies as a recipe if it contains a ``transform.mlir``.
    """
    names = [NONE_RECIPE]
    if _RECIPES_DIR.is_dir():
        for child in sorted(_RECIPES_DIR.iterdir()):
            if child.is_dir() and (child / "transform.mlir").is_file():
                names.append(child.name)
    return names


def resolve_recipe(name: str) -> Recipe | None:
    """Resolve a recipe name to its files, or ``None`` for the ``none`` sentinel.

    Args:
        name: Recipe name (must be in :func:`available_recipes`).

    Returns:
        A :class:`Recipe`, or ``None`` when ``name`` is ``"none"``.

    Raises:
        ValueError: If ``name`` is neither ``"none"`` nor a known recipe.
    """
    if name == NONE_RECIPE:
        return None
    recipe_dir = _RECIPES_DIR / name
    transform = recipe_dir / "transform.mlir"
    if not transform.is_file():
        raise ValueError(
            f"Unknown instrumentation recipe: '{name}'. "
            f"Choose from {available_recipes()}"
        )
    ip_dir = recipe_dir / "IPs"
    verilog = sorted(ip_dir.glob("*.v")) if ip_dir.is_dir() else []
    c_stubs = sorted(ip_dir.glob("*.c")) if ip_dir.is_dir() else []
    module_lib = ip_dir / "module_lib.xml"
    constraints = ip_dir / "constraints_STD.xml"
    return Recipe(
        name=name,
        transform_mlir=transform,
        ip_dir=ip_dir,
        verilog_inputs=verilog,
        c_excludes=c_stubs,
        module_lib=module_lib if module_lib.is_file() else None,
        constraints=constraints if constraints.is_file() else None,
    )
