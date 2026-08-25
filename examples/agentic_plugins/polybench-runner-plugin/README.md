# PolyBench Runner Plugin

A Claude Code plugin that orchestrates the SODA PolyBench HLS optimization workflow using specialized agents and skills for MLIR kernel preparation, planning, implementation, and synthesis.

Version of soda-benchmarks used: 0c3ba4fef31cb5b68e566373fd337731aeefc88e
## Prerequisites

- Claude Code installed with a supported API key in your environment
- The soda-benchmarks Dev Container (provides `soda-opt`, `bambu`, and other required binaries)
- [pixi](https://pixi.sh) for dependency management


## Running a Workflow

The plugin includes a Python workflow script that drives Claude Code agents to run synthesis experiments on PolyBench kernels.

### Usage

```bash
python examples/agentic_plugins/polybench-runner-plugin/scripts/workflow.py \
    --kernel threemm \
    --dimension TEST \
    --target Transformed
```

### Arguments

| Argument      | Required | Description                                         |
|---------------|----------|-----------------------------------------------------|
| `--kernel`    | Yes      | Kernel name (e.g. `threemm`, `gemm`, `mvt`)         |
| `--dimension` | Yes      | Dataset size (`TEST`, `MINI`, `SMALL`, `MEDIUM`, `LARGE`) |
| `--target`    | Yes      | One of `Baseline` or `Transformed`    |
| `--version`   | No       | Version label (default: `benchmark`)                |

### Targets

- **Baseline** -- Scaffolds an experiment and runs synthesis without any transformations.
- **Transformed** -- Scaffolds an experiment, plans and applies linalg tiling and affine unrolling optimizations, then runs synthesis.

