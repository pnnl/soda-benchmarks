"""Entry point for sb-cli: benchmark experiment scaffolding CLI.

Usage:
    python -m sb_cli <subcommand> [args]
    sb-cli <subcommand> [args]  # after pip install -e .

Runs from any working directory: the experiments location comes from the
imported `benches` package, not from where the command was invoked.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import benches

_DATASET_CHOICES = ["MINI", "SMALL", "MEDIUM", "LARGE", "EXTRALARGE"]
_DTYPE_CHOICES = ["float16", "float32", "float64"]
_TARGET_CHOICES = ["verilog", "optimized", "transformed", "gds", "llvm"]


def _common_parser() -> argparse.ArgumentParser:
    """Return a parent parser with the options shared by every subcommand."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--base_dir",
        default=None,
        help="Directory holding experiments/ (default: the benches/ package)",
    )
    return p


def _add_init_parser(
    subparsers: argparse._SubParsersAction, common: argparse.ArgumentParser
) -> None:
    p = subparsers.add_parser(
        "init", help="Scaffold a new experiment folder", parents=[common]
    )
    p.add_argument(
        "--output_dir",
        required=True,
        help="Logical name for the experiment (symlink name)",
    )
    p.add_argument(
        "--benchmark_name",
        default=None,
        help=(
            "Benchmark short name (e.g. gemm) or dotted import path "
            "(e.g. PolyBenchPyTorch.linear_algebra.blas.gemm)"
        ),
    )
    p.add_argument("--dataset", default="MINI", choices=_DATASET_CHOICES)
    p.add_argument("--dtype", default="float32", choices=_DTYPE_CHOICES)
    p.add_argument("--device", default="nangate45")
    p.add_argument("--clock_period", default=5.0, type=float)
    p.add_argument("--memory_policy", default="")
    p.add_argument("--target", default="verilog", choices=_TARGET_CHOICES)


def _add_fork_parser(
    subparsers: argparse._SubParsersAction, common: argparse.ArgumentParser
) -> None:
    p = subparsers.add_parser(
        "fork", help="Fork an existing experiment", parents=[common]
    )
    p.add_argument(
        "--from",
        dest="from_name_or_path",
        required=True,
        help="Registered name or path to source experiment",
    )
    p.add_argument(
        "--output_dir",
        required=True,
        help="Logical name for the new experiment",
    )


def _add_collect_parser(
    subparsers: argparse._SubParsersAction, common: argparse.ArgumentParser
) -> None:
    p = subparsers.add_parser(
        "collect", help="Collect metrics from experiments", parents=[common]
    )
    p.add_argument(
        "--from",
        dest="from_name_or_path",
        default=None,
        help="Registered name or path (omit to collect from all)",
    )


def _add_list_parser(
    subparsers: argparse._SubParsersAction, common: argparse.ArgumentParser
) -> None:
    subparsers.add_parser(
        "list", help="List the benchmarks available to --benchmark_name"
    )


def _resolve_base_dir(explicit: str | None) -> Path:
    """Return the directory holding experiments/ and registry.py.

    Defaults to the imported `benches` package, so sb-cli behaves the same from
    any working directory. `--base_dir` overrides it, which is also the seam for
    keeping experiments outside the checkout.

    Args:
        explicit: Value of --base_dir, or None.

    Returns:
        Absolute path to the base directory.
    """
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    return benches.ROOT


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="sb-cli",
        description="Benchmark experiment scaffolding and collection CLI",
    )
    common = _common_parser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_init_parser(subparsers, common)
    _add_fork_parser(subparsers, common)
    _add_collect_parser(subparsers, common)
    _add_list_parser(subparsers, common)

    args = parser.parse_args()

    if args.command == "list":
        from benches import catalog

        for name, bench in catalog.available().items():
            print(f"{name:12s} {bench.module}")
        return

    base_dir = _resolve_base_dir(args.base_dir)

    if args.command == "init":
        from sb_cli.flow import ExperimentConfig
        from sb_cli.init import scaffold

        config = ExperimentConfig(
            benchmark_name=args.benchmark_name,
            dataset=args.dataset,
            dtype=args.dtype,
            device=args.device,
            clock_period=args.clock_period,
            memory_policy=args.memory_policy,
            target=args.target,
        )
        scaffold(config, args.output_dir, base_dir)

    elif args.command == "fork":
        from sb_cli.fork import fork_experiment

        fork_experiment(args.from_name_or_path, args.output_dir, base_dir)

    elif args.command == "collect":
        from sb_cli.collect import collect_command

        collect_command(args.from_name_or_path, base_dir)


if __name__ == "__main__":
    main()
