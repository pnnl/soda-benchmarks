"""Flow configuration dataclass and target resolution for sb-cli.

A build target is described by three orthogonal axes rather than a single
opaque name:

* **flow** — how the MLIR is optimized (baseline, optimized, transformed)
* **backend** — what consumes the LLVM IR (bambu today; cpu/gpu reserved)
* **stage** — how far down the compilation path to go (llvm, verilog,
  simulation, gds)

`resolve_target` turns a triple into the Makefile TARGET path, which mirrors
the directory layout the mkinc rules already use (`bambu/<flow>/<artifact>`).
"""

from __future__ import annotations

from dataclasses import dataclass

from sb_cli.recipes import NONE_RECIPE, Recipe

FLOWS: tuple[str, ...] = ("baseline", "optimized", "transformed")
BACKENDS: tuple[str, ...] = ("bambu",)  # "cpu", "gpu" reserved for future use
STAGES: tuple[str, ...] = ("llvm", "verilog", "simulation", "gds")

# Default Bambu top function name. It is also baked into ll_to_verilog.sh and
# the soda_to_llvm_*.sh scripts, so it is a default here rather than a knob.
DEFAULT_TOP_FNAME = "forward_kernel"

# Makefile TARGET path per (backend, stage). The $(ODIR) prefix is not expanded
# here; it is substituted literally into the Makefile template, where ODIR is
# defined. Only the gds path uses {platform} and {top_fname}.
_TARGET_TEMPLATES: dict[tuple[str, str], str] = {
    ("bambu", "llvm"): "$(ODIR)/05_llvm_{flow}.ll",
    ("bambu", "verilog"): "$(ODIR)/bambu/{flow}/06_verilog.v",
    ("bambu", "simulation"): "$(ODIR)/bambu/{flow}/07_results.txt",
    ("bambu", "gds"): (
        "$(ODIR)/bambu/{flow}/HLS_output/Synthesis/bash_flow"
        "/openroad/results/{platform}/{top_fname}/base/6_final.gds"
    ),
}


def gds_platform(device: str) -> str:
    """Return the OpenROAD PDK directory name for a Bambu device.

    Bambu device names may carry a corner suffix (`asap7-BC`) that the OpenROAD
    results directory does not use, so the suffix is stripped.

    Args:
        device: Bambu device name, e.g. `nangate45` or `asap7-BC`.

    Returns:
        The bare PDK name used as a path component under `openroad/results/`.
    """
    return device.split("-")[0]


def _supported_stages(backend: str) -> list[str]:
    """Return the stages `backend` has a target template for, in STAGES order."""
    return [s for s in STAGES if (backend, s) in _TARGET_TEMPLATES]


def resolve_target(
    flow: str,
    backend: str,
    stage: str,
    *,
    device: str,
    top_fname: str = DEFAULT_TOP_FNAME,
) -> str:
    """Return the Makefile TARGET path for a (flow, backend, stage) triple.

    Args:
        flow: One of FLOWS.
        backend: One of BACKENDS.
        stage: One of STAGES.
        device: Bambu device name, used to derive the gds PDK directory.
        top_fname: Bambu top function name, used in the gds path.

    Returns:
        The TARGET path, with `$(ODIR)` left unexpanded.

    Raises:
        ValueError: If an axis value is unknown, or if the backend does not
            support the requested stage.
    """
    if flow not in FLOWS:
        raise ValueError(f"Unknown flow: '{flow}'. Choose from {list(FLOWS)}")
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend: '{backend}'. Choose from {list(BACKENDS)}")
    if stage not in STAGES:
        raise ValueError(f"Unknown stage: '{stage}'. Choose from {list(STAGES)}")

    template = _TARGET_TEMPLATES.get((backend, stage))
    if template is None:
        raise ValueError(
            f"Backend '{backend}' does not support stage '{stage}'. "
            f"Supported stages for '{backend}': {_supported_stages(backend)}"
        )
    return template.format(
        flow=flow, platform=gds_platform(device), top_fname=top_fname
    )


@dataclass
class ExperimentConfig:
    """All parameters for a single hardware synthesis experiment."""

    benchmark_name: str | None
    dataset: str
    dtype: str
    device: str
    clock_period: float
    memory_policy: str
    flow: str
    backend: str
    stage: str
    instrumentation: str = NONE_RECIPE  # recipe name; "none" = no instrumentation

    @property
    def target_name(self) -> str:
        """Return the human-readable target triple, e.g. `bambu/baseline/verilog`."""
        return f"{self.backend}/{self.flow}/{self.stage}"

    def target_path(self) -> str:
        """Return the Makefile TARGET path for this config."""
        return resolve_target(self.flow, self.backend, self.stage, device=self.device)


def ip_integration_block(recipe: Recipe | None) -> str:
    """Render the Bambu IP-integration Makefile block for a recipe.

    Returns an empty string for ``None`` (the ``none`` recipe), so a
    non-instrumented experiment's Makefile carries no IP wiring. Otherwise it
    emits the ``BAMBU_IP_INTEGRATION`` / ``IP_*`` / ``EXTRA_VERILOG_DEPS``
    variables consumed by ``scripts/mkinc/llvm_to_verilog.mk`` and
    ``scripts/ll_to_verilog.sh``, referencing the IP files copied into the
    experiment's ``IPs/`` directory. Paths use ``$(IPDIR)`` so the block is
    relocatable with the experiment folder.
    """
    if recipe is None:
        return ""

    # Comma-separated lists (full paths, per llvm_to_verilog.mk convention).
    verilog = ",".join(f"$(IPDIR)/{p.name}" for p in recipe.verilog_inputs)
    c_exclude = ",".join(f"$(IPDIR)/{p.name}" for p in recipe.c_excludes)
    module_lib = "$(IPDIR)/module_lib.xml" if recipe.module_lib else ""
    constraints = "$(IPDIR)/constraints_STD.xml" if recipe.constraints else ""

    return (
        f"# IP integration flow (instrumentation recipe: {recipe.name})\n"
        "EXP_CWD:=$(CURDIR)\n"
        "IPDIR=$(EXP_CWD)/IPs\n"
        "BAMBU_IP_INTEGRATION=true\n"
        "# Comma-separated lists of files, with full paths.\n"
        f"IP_C_EXCLUDE={c_exclude}\n"
        f"IP_VERILOG_INPUTS={verilog}\n"
        f"IP_MODULE_LIB={module_lib}\n"
        f"IP_CONSTRAINTS={constraints}\n"
        "# Space-separated list for Makefile dependencies.\n"
        "IP_INTEGRATION_FILES := \\\n"
        "  $(IP_C_EXCLUDE) \\\n"
        "  $(IP_VERILOG_INPUTS) \\\n"
        "  $(IP_MODULE_LIB) \\\n"
        "  $(IP_CONSTRAINTS)\n"
        "EXTRA_VERILOG_DEPS := $(IP_INTEGRATION_FILES)\n"
    )
