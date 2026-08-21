---
name: soda-benchmarks
description: Orchestrates the SODA PolyBench HLS optimization workflow. Use when
  the user asks to run a benchmark, scaffold an experiment, or execute the full
  Transformed/Baseline/Optimized pipeline for a PolyBench kernel. Trigger phrases
  include "run benchmark", "run experiment", "scaffold experiment", "run workflow",
  "baseline", "transformed", "optimized".
---

# SODA PolyBench Workflow

Orchestrate the SODA PolyBench HLS optimization workflow for a given kernel, dimension, and target.

## Parameters

- **Kernel**: The PolyBench kernel name (e.g. `threemm`, `gemm`, `mvt`)
- **Target**: One of `Baseline`, `Transformed`, or `Optimized`
- **Dimension**: Dataset size (e.g. `TEST`, `MINI`, `SMALL`, `MEDIUM`, `LARGE`)

## Workflow: Transformed

1. Scaffold experiment using `experiment-preparation` agent in `/benches/experiments` folder. Name output directory `<kernel>_<dimension>_<target>`.
2. Analyze `02_linalg_tagged.mlir` using `mlir-linalg-hls-planner` agent.
3. Implement linalg HLS plan using `mlir-linalg-implementer` agent. Ensure tiling transformation schedule is named `02_linalg_tile_ts.mlir`.
4. Use `Skill(mlir-annotator)` to annotate `04_affine_tiled` into `05_affine_tagged`.
5. Analyze `05_affine_tagged` using `mlir-affine-hls-planner` agent.
6. Implement affine HLS plan using `mlir-affine-implementer` agent. Ensure unrolling transformation schedule is named `04_affine_unroll_ts.mlir`.
7. Use `experiment-runner` agent to run transformed simulation.
8. Use `Skill(bambu-log-parser)` to generate JSON at: `experiment_directory/transformation/bambu_summary.json`.

**Important:** Ensure that the linalg HLS plan and affine HLS plan are saved to the `experiment_directory/transformation` directory.

## Workflow: Baseline

1. Scaffold experiment using `experiment-preparation` agent in `/benches/experiments` folder. Name output directory `<kernel>_<dimension>_<target>`.
2. Use `experiment-runner` agent to run baseline simulation.
3. Use `Skill(bambu-log-parser)` to generate JSON at: `experiment_directory/transformation/bambu_summary.json`.

## Workflow: Optimized

1. Scaffold experiment using `experiment-preparation` agent in `/benches/experiments` folder. Name output directory `<kernel>_<dimension>_<target>`.
2. Use `experiment-runner` agent to run optimized simulation.
3. Use `Skill(bambu-log-parser)` to generate JSON at: `experiment_directory/transformation/bambu_summary.json`.

## Error Handling

On error: stop and report. Do not attempt fixes.
