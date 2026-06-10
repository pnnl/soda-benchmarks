"""Flow configuration dataclass and TARGET_MAP for sb-cli.

Defines the ExperimentConfig dataclass (used by init/fork) and the
TARGET_MAP mapping human-readable target names to Makefile TARGET paths.
"""

from __future__ import annotations

from dataclasses import dataclass

# Maps human-readable target names to Makefile TARGET path values.
# The $(ODIR) prefix is not expanded here; it is substituted literally
# into the Makefile template as the ODIR variable is defined there.
TARGET_MAP: dict[str, str] = {
    "verilog": "$(ODIR)/bambu/baseline/06_verilog.v",
    "optimized": "$(ODIR)/bambu/optimized/06_verilog.v",
    "transformed": "$(ODIR)/bambu/transformed/06_verilog.v",
    "gds": (
        "$(ODIR)/bambu/baseline/HLS_output/Synthesis/bash_flow"
        "/openroad/results/nangate45/forward_kernel/base/6_final.gds"
    ),
    "llvm": "$(ODIR)/04_llvm.ll",
}


@dataclass
class ExperimentConfig:
    """All parameters for a single hardware synthesis experiment."""

    benchmark_name: str | None
    dataset: str
    dtype: str
    device: str
    clock_period: float
    memory_policy: str
    target: str  # human name — resolved via TARGET_MAP

    def target_path(self) -> str:
        """Return the Makefile TARGET path for this config's target name."""
        if self.target not in TARGET_MAP:
            raise ValueError(
                f"Unknown target: '{self.target}'. Choose from {list(TARGET_MAP)}"
            )
        return TARGET_MAP[self.target]
