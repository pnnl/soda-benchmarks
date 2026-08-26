# Agentic Polybench Runner

This guide will install a claude plugin that allows an orchestrator to optimize PyTorch PolyBench kernels and run synthesis experiments using the `sb-cli` tool.

To have Claude Code agents run a synthesis experiment, use the
[PolyBench Runner Plugin](examples/agentic_plugins/polybench-runner-plugin/):

1. Ensure you have Claude Code installed and a supported API key in your environment
2. Run the workflow script, specifying the kernel, dataset dimension, and optimization target:

```bash
python examples/agentic_plugins/polybench-runner-plugin/scripts/workflow.py \
    --kernel threemm \
    --dimension TEST \
    --target Transformed
```

Available targets are `Baseline` (no transformations), `Transformed` (linalg
tiling + affine unrolling).
Results are logged to MLflow.