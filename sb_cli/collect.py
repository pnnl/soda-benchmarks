"""sb-cli collect command: gather metrics from experiment output directories.

Uses an extensible MetricCollector architecture. Each collector registers
a metric name, a glob pattern, and a parser callable. Results are written
to output/metrics.json in each experiment directory.

The sink abstraction enables future MLflow backend swap without
restructuring collector logic.
"""

from __future__ import annotations

import json
import re
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


def _read_text(path: Path) -> str | None:
    """Read a text file, returning None (never raising) if it is unreadable."""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


# Matches Bambu's simulation summary line, e.g.:
#   Total cycles             : 14028 cycles
_CYCLE_COUNT_RE = re.compile(r"Total cycles\s*:\s*(\d+)\s*cycles")

# Matches per-function flip-flop totals, e.g.:
#   Total number of flip-flops in function forward_kernel: 9911
_FLIP_FLOPS_RE = re.compile(r"Total number of flip-flops in function (\S+):\s*(\d+)")

# Matches per-function area estimates, e.g.:
#   Total estimated area: 885472
_ESTIMATED_AREA_RE = re.compile(r"Total estimated area:\s*(\d+)")

# Matches "Summary of resources" cell totals, e.g.:
#   Total cells    : 4334
_TOTAL_CELLS_RE = re.compile(r"Total cells\s*:\s*(\d+)")

# Matches HW performance-counter prints, e.g.:
#   [HW] sodaInstrHWCounters: location 0000000000000004 count        106
_HW_COUNTER_RE = re.compile(
    r"\[HW\] sodaInstrHWCounters: location ([0-9a-fA-F]+) count\s+(\d+)"
)

# Matches the SW C-stub start/stop markers, e.g.:
#   [SW] HW counter STARTED at loc: 4
#   [SW] HW counter STOPED at loc: 4
_SW_COUNTER_RE = re.compile(r"\[SW\] HW counter (STARTED|STOPED) at loc: (\d+)")

# Matches assertion checker prints, e.g.:
#   sodaInstrAssertLessThen: 0000000000000003 < 000000000000000a ? true
_ASSERT_RE = re.compile(
    r"sodaInstrAssertLessThen: [0-9a-fA-F]+ < [0-9a-fA-F]+ \? (true|false)"
)


def _parse_cycle_count(path: Path) -> int | None:
    """Extract the Bambu Verilator simulation cycle count from a Bambu log.

    Looks for a line such as ``Total cycles             : 14028 cycles``
    (emitted at the end of ``output/bambu/<variant>/bambu-log``).

    Returns:
        The cycle count as an int, or None if the file is missing/unreadable
        or does not contain the expected line.
    """
    text = _read_text(path)
    if text is None:
        return None
    match = _CYCLE_COUNT_RE.search(text)
    return int(match.group(1)) if match else None


def _parse_resource_usage(path: Path) -> dict[str, Any]:
    """Extract resource-usage figures (flip-flops, area, cells) from a Bambu log.

    Parses the "Summary of resources"-adjacent lines that Bambu prints once
    per synthesized function, e.g.::

        Total estimated area: 885472
        Total number of flip-flops in function forward_kernel: 9911
        Total cells    : 4334

    The *last* flip-flop/area entries in the log correspond to the top-level
    kernel function (helper/library functions are reported earlier), so those
    are surfaced as ``top_function*`` keys alongside the full per-function
    breakdown.

    Returns:
        A dict with per-function flip-flop counts, the raw list of area
        estimates, and top-function/summary figures. Empty dict if the file
        is missing/unreadable.
    """
    text = _read_text(path)
    if text is None:
        return {}

    flip_flops_by_function = {
        name: int(count) for name, count in _FLIP_FLOPS_RE.findall(text)
    }
    estimated_area_values = [int(area) for area in _ESTIMATED_AREA_RE.findall(text)]
    total_cells_values = [int(cells) for cells in _TOTAL_CELLS_RE.findall(text)]

    result: dict[str, Any] = {
        "flip_flops_by_function": flip_flops_by_function,
        "estimated_area_values": estimated_area_values,
    }
    if flip_flops_by_function:
        top_function, top_flip_flops = list(flip_flops_by_function.items())[-1]
        result["top_function"] = top_function
        result["top_function_flip_flops"] = top_flip_flops
    if estimated_area_values:
        result["top_function_estimated_area"] = estimated_area_values[-1]
    if total_cells_values:
        result["total_cells"] = total_cells_values[-1]
    return result


def _parse_instrumentation_events(path: Path) -> dict[str, Any]:
    """Summarize instrumentation ``$display`` events from a Bambu log.

    Covers three event families emitted by the instrumentation IPs during
    simulation:

    - HW performance counters: ``[HW] sodaInstrHWCounters: location <hex>
      count <n>`` — the final (last-printed) count per location is kept.
    - SW C-stub markers: ``[SW] HW counter STARTED/STOPED at loc: <n>``.
    - Assertion checker: ``sodaInstrAssertLessThen: <hex> < <hex> ?
      true/false``.

    Returns:
        A dict with per-location final counter values, SW start/stop event
        counts, and assertion pass/fail counts. Empty dict if the file is
        missing/unreadable.
    """
    text = _read_text(path)
    if text is None:
        return {}

    hw_counter_final_count_by_location: dict[str, int] = {}
    for location_hex, count in _HW_COUNTER_RE.findall(text):
        location = str(int(location_hex, 16))
        hw_counter_final_count_by_location[location] = int(count)

    sw_counter_event_counts = {"STARTED": 0, "STOPED": 0}
    for state, _loc in _SW_COUNTER_RE.findall(text):
        sw_counter_event_counts[state] += 1

    assertion_results = {"true": 0, "false": 0}
    for outcome in _ASSERT_RE.findall(text):
        assertion_results[outcome] += 1

    return {
        "hw_counter_final_count_by_location": hw_counter_final_count_by_location,
        "sw_counter_event_counts": sw_counter_event_counts,
        "assertion_results": assertion_results,
        "assertion_total": sum(assertion_results.values()),
    }


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
    MetricCollector(
        name="simulation_cycles",
        pattern="output/bambu/**/bambu-log",
        parser=_parse_cycle_count,
    ),
    MetricCollector(
        name="resource_usage",
        pattern="output/bambu/**/bambu-log",
        parser=_parse_resource_usage,
    ),
    MetricCollector(
        name="instrumentation_events",
        pattern="output/bambu/**/bambu-log",
        parser=_parse_instrumentation_events,
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
