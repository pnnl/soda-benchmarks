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

# Files only some experiments have, copied when present and not warned about
# when absent: sc_flow.py exists only for --builder siliconcompiler.
_OPTIONAL_FILES = [
    "sc_flow.py",
]


def fork_experiment(
    from_name_or_path: str, output_dir: str | None, base_dir: Path
) -> Path:
    """Fork an existing experiment into a new directory.

    Args:
        from_name_or_path: Registered experiment name or filesystem path.
        output_dir: Logical name for the new experiment. When None, it is
            derived from the source name with a `-NNN` counter appended.
        base_dir: Base directory (benches/).

    Returns:
        Path to the new timestamped experiment directory.
    """
    from sb_cli.init import _timestamp, next_available_name

    registry = Registry(base_dir)

    print(f"[sb-cli] Forking from: {from_name_or_path} (resolving...)")
    source_dir = registry.resolve(from_name_or_path, base_dir)

    # Determine whether it was found in registry
    experiments = registry.load()
    from_registry = from_name_or_path in experiments
    if from_registry:
        print(
            f"[sb-cli] Forking from: {from_name_or_path} "
            f"(resolved via registry → {source_dir})"
        )
    else:
        print(f"[sb-cli] Forking from: {source_dir} (resolved via path)")

    # Named after the source, so repeated forks of one experiment form a series.
    # Done before anything is created, so a failure leaves no empty directory.
    if output_dir is None:
        stem = from_name_or_path if from_registry else source_dir.name
        output_dir = next_available_name(stem, base_dir)
        print(f"[sb-cli] Auto-named fork: {output_dir}")

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

    # Copy the files an experiment may or may not have, preserving the mode so
    # an executable sc_flow.py stays executable in the fork.
    for fname in _OPTIONAL_FILES:
        src = source_dir / fname
        if src.exists():
            shutil.copy2(src, new_dir / fname)

    # Carry the instrumentation IP directory (present when the source was
    # scaffolded with an --instrumentation recipe) so the fork stays buildable.
    ip_src = source_dir / "IPs"
    if ip_src.is_dir():
        shutil.copytree(ip_src, new_dir / "IPs")

    rel_path = f"experiments/{ts}"
    registry.append(output_dir, rel_path)

    symlink_path = (base_dir / "experiments" / output_dir).absolute()
    print(f"[sb-cli] Created experiment: {new_dir.absolute()}")
    print(f"[sb-cli] Symlink: {symlink_path} -> {ts}/")
    print(f"[sb-cli] Registered as '{output_dir}' in experiments/registry.py")
    return new_dir
