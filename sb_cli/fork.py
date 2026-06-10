"""sb-cli fork command: create a new experiment by copying an existing one.

Hard-copies all git-tracked config files from the source experiment into
a new timestamped directory, with a clean output/ (not copied).
"""

from __future__ import annotations

import shutil
from pathlib import Path

from sb_cli.registry import Registry

# Files copied from source experiment (git-tracked, not output/)
_TRACKED_FILES = [
    "torchscript.py",
    "flow.py",
    "Makefile",
    "transform.mlir",
    "README.md",
    ".gitignore",
]


def fork_experiment(from_name_or_path: str, output_dir: str, base_dir: Path) -> Path:
    """Fork an existing experiment into a new directory.

    Args:
        from_name_or_path: Registered experiment name or filesystem path.
        output_dir: Logical name for the new experiment.
        base_dir: Base directory (benches/).

    Returns:
        Path to the new timestamped experiment directory.
    """
    from sb_cli.init import _timestamp

    registry = Registry(base_dir)

    print(f"[sb-cli] Forking from: {from_name_or_path} (resolving...)")
    source_dir = registry.resolve(from_name_or_path, base_dir)

    # Determine whether it was found in registry
    experiments = registry.load()
    if from_name_or_path in experiments:
        print(
            f"[sb-cli] Forking from: {from_name_or_path} "
            f"(resolved via registry → {source_dir})"
        )
    else:
        print(f"[sb-cli] Forking from: {source_dir} (resolved via path)")

    ts = _timestamp(base_dir)
    new_dir = base_dir / "experiments" / ts
    new_dir.mkdir(parents=True, exist_ok=True)

    # Check symlink target does not already exist
    from sb_cli.init import _create_symlink
    _create_symlink(new_dir, output_dir, base_dir)

    # Hard-copy tracked files
    for fname in _TRACKED_FILES:
        src = source_dir / fname
        dst = new_dir / fname
        if src.exists():
            shutil.copy2(src, dst)
        else:
            print(f"[sb-cli] WARNING: {fname} not found in source, skipping")

    rel_path = f"experiments/{ts}"
    registry.append(output_dir, rel_path)

    print(f"[sb-cli] Created experiment: experiments/{ts}/")
    print(f"[sb-cli] Symlink: experiments/{output_dir} -> {ts}/")
    print(f"[sb-cli] Registered as '{output_dir}' in experiments/registry.py")
    return new_dir
