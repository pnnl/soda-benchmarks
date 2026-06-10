"""sb-cli collect command: gather metrics from experiment output directories.

Uses an extensible MetricCollector architecture. Each collector registers
a metric name, a glob pattern, and a parser callable. Results are written
to output/metrics.json in each experiment directory.

The sink abstraction enables future MLflow backend swap without
restructuring collector logic.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from sb_cli.registry import Registry


@dataclass
class MetricCollector:
    """A single metric extraction unit.

    Attributes:
        name: Metric name (key in metrics.json output).
        pattern: Glob pattern relative to experiment root (e.g. 'output/*.ll').
        parser: Callable taking a matched Path and returning a metric value.
    """

    name: str
    pattern: str
    parser: Callable[[Path], Any]


def _count_lines(path: Path) -> int:
    """Count lines in a text file."""
    try:
        return sum(1 for _ in path.open(encoding="utf-8", errors="replace"))
    except OSError:
        return 0


def _file_info(path: Path) -> dict[str, Any]:
    """Return file size and existence as a dict."""
    try:
        stat = path.stat()
        return {"size_bytes": stat.st_size, "exists": True}
    except OSError:
        return {"size_bytes": 0, "exists": False}


# Built-in collectors shipped with sb-cli
BUILTIN_COLLECTORS: list[MetricCollector] = [
    MetricCollector(
        name="ll_line_count",
        pattern="output/*.ll",
        parser=_count_lines,
    ),
    MetricCollector(
        name="v_line_count",
        pattern="output/bambu/**/*.v",
        parser=_count_lines,
    ),
    MetricCollector(
        name="file_inventory",
        pattern="output/**/*",
        parser=_file_info,
    ),
]


def _write_metrics_json(experiment_path: Path, data: dict[str, Any]) -> None:
    """Default sink: write metrics dict to output/metrics.json.

    Args:
        experiment_path: Path to the experiment directory.
        data: Metrics data to write.
    """
    output_dir = experiment_path / "output"
    output_dir.mkdir(exist_ok=True)
    out_path = output_dir / "metrics.json"
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


class CollectionDriver:
    """Runs registered MetricCollectors against an experiment directory."""

    def __init__(
        self,
        collectors: list[MetricCollector] | None = None,
        sink: Callable[[Path, dict[str, Any]], None] | None = None,
    ) -> None:
        self.collectors = collectors if collectors is not None else BUILTIN_COLLECTORS
        self.sink = sink if sink is not None else _write_metrics_json

    def run(self, experiment_path: Path) -> dict[str, Any]:
        """Collect metrics from experiment_path and call sink.

        Args:
            experiment_path: Absolute path to the experiment directory.

        Returns:
            Full metrics dict written by the sink.
        """
        metrics: dict[str, Any] = {}

        for collector in self.collectors:
            matched = list(experiment_path.glob(collector.pattern))
            # For file_inventory, only include files (not directories)
            if collector.name == "file_inventory":
                matched = [p for p in matched if p.is_file()]
                result = {
                    str(p.relative_to(experiment_path)): collector.parser(p)
                    for p in matched
                }
            elif matched:
                result = {
                    str(p.relative_to(experiment_path)): collector.parser(p)
                    for p in matched
                }
            else:
                result = {}
            metrics[collector.name] = result

        data: dict[str, Any] = {
            "collected_at": datetime.now().isoformat(timespec="seconds"),
            "experiment": experiment_path.name,
            "metrics": metrics,
        }
        self.sink(experiment_path, data)
        return data


def collect_command(name_or_path: str | None, base_dir: Path) -> None:
    """Run collection on one or all registered experiments.

    Args:
        name_or_path: Registered name or path, or None for all.
        base_dir: Base directory (benches/).
    """
    registry = Registry(base_dir)
    driver = CollectionDriver()

    if name_or_path is not None:
        targets = [registry.resolve(name_or_path, base_dir)]
    else:
        experiments = registry.load()
        targets = []
        for name, rel_path in experiments.items():
            candidate = Path(rel_path)
            if not candidate.is_absolute():
                candidate = base_dir / candidate
            candidate = candidate.resolve()
            if candidate.is_dir():
                targets.append(candidate)
            else:
                print(
                    f"[sb-cli] WARNING: Skipping '{name}' "
                    f"— directory not found: {candidate}"
                )

    for exp_path in targets:
        print(f"[sb-cli] Collecting from: {exp_path}")
        data = driver.run(exp_path)
        metrics_path = exp_path / "output" / "metrics.json"
        for cname, cdata in data["metrics"].items():
            count = len(cdata)
            print(f"[sb-cli]   {cname}: {count} file(s) matched")
        print(f"[sb-cli]   Written: {metrics_path}")
