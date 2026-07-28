"""sb-cli init command: scaffold a new experiment folder.

Creates a timestamped experiment directory under experiments/, generates
all required files from templates, creates a named symlink, and registers
the experiment in experiments/registry.py.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from benches import catalog
from sb_cli.flow import ExperimentConfig
from sb_cli.registry import Registry
from sb_cli.templates import render

# Files written into each experiment directory
_GENERATED_FILES = [
    "torchscript.py",
    "flow.py",
    "Makefile",
    "transform.mlir",
    "README.md",
    ".gitignore",
]


def _timestamp(base_dir: Path) -> str:
    """Return a unique timestamp string in YYYY_MM_DD_HH_MM_SS[_N] format.

    Appends a counter suffix if the experiments/ directory for this second
    already exists, ensuring uniqueness within the same second.
    """
    ts = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    candidate = base_dir / "experiments" / ts
    if not candidate.exists():
        return ts
    counter = 1
    while (base_dir / "experiments" / f"{ts}_{counter}").exists():
        counter += 1
    return f"{ts}_{counter}"


def _create_experiment_dir(timestamp: str, base_dir: Path) -> Path:
    """Create and return the timestamped experiment directory.

    Args:
        timestamp: Timestamp string used as directory name.
        base_dir: Base directory (benches/).

    Returns:
        Path to the newly created timestamped directory.
    """
    exp_dir = base_dir / "experiments" / timestamp
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def _create_symlink(ts_dir: Path, name: str, base_dir: Path) -> None:
    """Create a relative symlink experiments/<name> -> <timestamp>/.

    Args:
        ts_dir: Absolute path to the timestamped experiment directory.
        name: Logical name for the symlink.
        base_dir: Base directory (benches/).

    Raises:
        SystemExit: If a symlink with this name already exists.
    """
    symlink_path = base_dir / "experiments" / name
    if symlink_path.exists() or symlink_path.is_symlink():
        print(
            f"[sb-cli] ERROR: '{symlink_path}' already exists. "
            "Choose a different --output_dir."
        )
        raise SystemExit(1)
    # Create a relative symlink (just the timestamp dir name)
    symlink_path.symlink_to(ts_dir.name)


def _resolve_benchmark(benchmark_name: str) -> catalog.Benchmark:
    """Look up `benchmark_name` in the benches catalog.

    Accepts a short name (`gemm`) or a dotted path with or without the
    `benches.` prefix; see `benches.catalog.find`.

    Args:
        benchmark_name: Benchmark name or dotted module path.

    Returns:
        The matching `Benchmark`.

    Raises:
        SystemExit: If no benchmark matches.
    """
    bench = catalog.find(benchmark_name)
    if bench is None:
        print(f"[sb-cli] ERROR: Unknown benchmark '{benchmark_name}'.")
        print(f"[sb-cli] Available: {', '.join(catalog.available())}")
        raise SystemExit(1)
    return bench


def scaffold(config: ExperimentConfig, output_dir: str, base_dir: Path) -> Path:
    """Create a complete experiment folder from config.

    Args:
        config: ExperimentConfig with all hardware and benchmark parameters.
        output_dir: Logical name for the experiment (symlink name).
        base_dir: Base directory (benches/).

    Returns:
        Path to the new timestamped experiment directory.
    """
    # Resolve the benchmark before creating any files
    bench: catalog.Benchmark | None = None
    if config.benchmark_name is not None:
        bench = _resolve_benchmark(config.benchmark_name)

    ts = _timestamp(base_dir)
    exp_dir = _create_experiment_dir(ts, base_dir)
    rel_path = f"experiments/{ts}"
    target_path = config.target_path()
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Build template context
    ctx: dict[str, object] = {
        "experiment_name": output_dir,
        "benchmark_name": config.benchmark_name or "None",
        "benchmark_name_repr": repr(config.benchmark_name),
        "dataset": config.dataset,
        "dtype": config.dtype,
        "device": config.device,
        "clock_period": config.clock_period,
        "memory_policy": config.memory_policy,
        "target_name": config.target,
        "target_path": target_path,
        "created_at": created_at,
    }

    # Render torchscript.py
    if bench is not None:
        ctx["kernel_name"] = bench.name
        ctx["kernel_class"] = catalog.model_class(bench)
        ctx["kernel_module"] = bench.module
        ctx["utils_module"] = bench.utils_module
        torchscript_content = render("torchscript_bench.py.tmpl", ctx)
    else:
        torchscript_content = render("torchscript_default.py.tmpl", ctx)

    # Write all files
    (exp_dir / "torchscript.py").write_text(torchscript_content, encoding="utf-8")
    (exp_dir / "flow.py").write_text(render("flow.py.tmpl", ctx), encoding="utf-8")
    (exp_dir / "Makefile").write_text(render("Makefile.tmpl", ctx), encoding="utf-8")
    (exp_dir / "transform.mlir").write_text(
        render("transform.mlir.tmpl", ctx), encoding="utf-8"
    )
    (exp_dir / "README.md").write_text(render("README.md.tmpl", ctx), encoding="utf-8")
    (exp_dir / ".gitignore").write_text(render("gitignore.tmpl", ctx), encoding="utf-8")

    # Create symlink and register
    _create_symlink(exp_dir, output_dir, base_dir)
    Registry(base_dir).append(output_dir, rel_path)

    symlink_path = (base_dir / "experiments" / output_dir).absolute()
    print(f"[sb-cli] Created experiment: {exp_dir.absolute()}")
    print(f"[sb-cli] Symlink: {symlink_path} -> {ts}/")
    print(f"[sb-cli] Registered as '{output_dir}' in experiments/registry.py")
    return exp_dir
