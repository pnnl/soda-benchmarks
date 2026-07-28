# Benchmarks in PyTorch

These subfolders contains benchmark suites that were translated from other benchmarking projects and adapted to work with a PyTorch -> MLIR workflow.

This directory is part of the soda-benchmarks project and is managed by the
root-level `pixi.toml`/`pyproject.toml`. All `pixi run ...` commands below
should be run from the soda-benchmarks repo root
(`/workspaces/soda/soda-benchmarks`), not from within `benches/`.

# Purpose

Provide a small, focused collection of benchmark kernels and tooling showing how to convert PyTorch models to MLIR using the parent's MLIR/PyTorch tooling, and to scaffold hardware synthesis experiments from those kernels. The benchmarks are intended for research and performance experimentation.

# What you'll find here

- `PolyBenchC-4.2.1/` — the original PolyBench-C benchmark suite (read-only upstream copy). This contains many kernels organised by category (linear-algebra, stencils, medley, etc.).
- `PolyBenchPyTorch/` — (converted kernels) contains PyTorch-layer translations of selected PolyBench kernels
- `experiments/` — generated hardware synthesis experiments (see `sb-cli` below)

The `sb-cli` experiment scaffolding CLI itself lives at the repo root in
[`sb_cli/`](../sb_cli/), alongside the root `pixi.toml`/`pyproject.toml`.

For an end-to-end PyTorch -> Verilog/GDS example, see
[`examples/pytorch-to-gds/mm-no_weights/`](../examples/pytorch-to-gds/mm-no_weights/)
in the parent project.

# Notes for contributors and researchers

- The PolyBench-C tree under `PolyBenchC-4.2.1/` is included for reference and is not meant to be modified here; it preserves the upstream sources.
- Converted PyTorch versions of kernels live under `PolyBenchPyTorch/` — these are the files you should edit when experimenting with different layer implementations or benchmarking setups.

# Development

### Running the test suite

From the soda-benchmarks repo root:

```bash
pixi run test-fast    # forward-pass shape checks only (~1.5s)
pixi run test-mlir    # MLIR generation tests (~4s, requires torch-mlir)
pixi run test         # all tests
pixi run lint         # ruff check
pixi run format       # ruff format
```

### Adding a kernel to an existing suite

The existing suites are `PolyBenchPyTorch/linear_algebra/kernels/` and
`PolyBenchPyTorch/linear_algebra/blas/`. To add a kernel (e.g. `foo`) to
either:

1. **Create the kernel directory and files:**
   ```
   PolyBenchPyTorch/linear_algebra/<suite>/foo/
       __init__.py   # empty
       foo.py        # kernel implementation
   ```

2. **Implement `foo.py`** following the pattern of any existing kernel:
   - Import shared utilities from
     `benches.PolyBenchPyTorch.linear_algebra.utils`. Always spell imports with
     the `benches.` prefix — an unqualified `PolyBenchPyTorch...` only resolves
     when the interpreter starts inside `benches/`, and loads a second,
     unrelated copy of the same modules.
   - Define a `get_dataset_dimensions` function — 1-arg for `kernels/`,
     2-arg for `blas/` (see memory notes for the distinction).
   - Implement the kernel as an `nn.Module` subclass with a `forward()` method.
   - Implement `init_array()` returning all input tensors (scalars as
     0-dim `torch.tensor(val, dtype=dtype)` for torch-mlir compatibility).
   - Implement `parse_args()` and `main()` for standalone MLIR generation.

3. **Register dataset dimensions** in
   `PolyBenchPyTorch/linear_algebra/utils.py` — add an entry to `_DATASETS`
   keyed by the kernel name.

4. **Add tests** in `tests/test_kernels.py` and `tests/test_mlir.py`
   following the pattern of the existing 13 tests in each file.

5. **Verify:**
   ```bash
   pixi run test
   pixi run lint
   ```

### Adding a new benchmark suite entirely

A new suite is a top-level Python package sitting alongside `PolyBenchPyTorch/`
(e.g. `MLPerfPyTorch/`, `PolyBenchJAX/`). It is self-contained — its own
package tree, its own utils, its own dataset registry.

1. **Create the top-level package:**
   ```
   <SuiteName>/
       __init__.py
       utils.py          # dataset registry and shared helpers for this suite
       <category>/
           __init__.py
           <kernel>/
               __init__.py
               <kernel>.py
   ```

2. **Write `<SuiteName>/utils.py`** — at minimum:
   - `_DATASETS` dict mapping kernel names → dataset sizes → dimension dicts.
   - `get_dataset_dimensions(kernel, dataset)` lookup function.
   - `generate_mlir(model, inputs, out_path, dialect)` — can import and
     re-export the implementation from `PolyBenchPyTorch/linear_algebra/utils.py`
     to avoid duplication, or provide its own if the compilation pipeline differs.

3. **Implement kernels** following the same `nn.Module` + `init_array` +
   `main()` pattern as the existing suite. Scalars passed to `forward()` must
   be 0-dim tensors (`torch.tensor(val, dtype=dtype)`) for torch-mlir tracing.

4. **Add `<SuiteName>` to `sys.path`** — `tests/conftest.py` already adds the
   repo root, so any top-level package is importable in tests without further
   changes.

5. **Add tests** — create `tests/test_<suite>_kernels.py` and
   `tests/test_<suite>_mlir.py` mirroring the existing test files. Import
   kernels via `importlib.import_module("<SuiteName>.<category>.<kernel>")`.

6. **Update the lint and coverage config** (in the repo root `pixi.toml` /
   `pyproject.toml`):
   - In `pixi.toml`, add the new package to the lint/format tasks:
     ```toml
     lint = "ruff check benches/PolyBenchPyTorch benches/<SuiteName> benches/tests"
     format = "ruff format benches/PolyBenchPyTorch benches/<SuiteName> benches/tests"
     ```
   - In `pyproject.toml`, extend the coverage source list:
     ```toml
     [tool.coverage.run]
     source = ["benches/PolyBenchPyTorch", "benches/<SuiteName>"]
     ```

# sb-cli — Experiment Scaffolding CLI

`sb-cli` automates the creation of hardware synthesis experiment folders from
any PolyBench (or future benchmark) kernel. It can be run from any directory —
it locates `benches/experiments/` through the imported `benches` package, not
through the current working directory. Use `--base_dir` to point it elsewhere.

List the benchmarks it accepts with `pixi run sb-cli list`. `--benchmark_name`
takes either the short name (`gemm`) or the full dotted path:

```bash
pixi run sb-cli init \
  --benchmark_name PolyBenchPyTorch.linear_algebra.blas.gemm \
  --dataset MEDIUM \
  --dtype float32 \
  --device nangate45 \
  --clock_period 5 \
  --target verilog \
  --output_dir gemm_medium_baseline
```

### Scaffold a new experiment

This creates `experiments/gemm_medium_baseline/` containing `torchscript.py`,
`flow.py`, `Makefile`, `transform.mlir`, `README.md`, and `.gitignore`.

### Run the synthesis flow

```bash
cd experiments/gemm_medium_baseline
python flow.py       # sets BAMBU_* env vars and calls make
```

### Fork an experiment (create a variant)

```bash
pixi run sb-cli fork \
  --from gemm_medium_baseline \
  --output_dir gemm_medium_transformed

# Edit the new transform.mlir, then run:
cd experiments/gemm_medium_transformed && python flow.py
```

### Collect metrics

```bash
pixi run sb-cli collect --from gemm_medium_baseline   # single experiment
pixi run sb-cli collect                               # all registered experiments
```

Results are written to `experiments/<name>/output/metrics.json`.

---

# Common tasks

- List available kernels: `pixi run sb-cli list`.
- Run a single kernel standalone to produce MLIR output (from the repo root):
  ```bash
  pixi run python -m benches.PolyBenchPyTorch.linear_algebra.blas.gemm.gemm \
      output/gemm.mlir --dialect tosa --dataset SMALL --dtype float32
  ```

# Troubleshooting

- If `import benches` or `import sb_cli` fails, run `pixi install`. Both come
  from an editable install of this repo (declared in `pyproject.toml` and
  `pixi.toml`'s `[pypi-dependencies]`) — they are deliberately **not** on
  `PYTHONPATH`, which is reserved for dependencies installed outside the pixi
  environment.
- If MLIR generation fails while importing MLIR/PyTorch conversion bindings, verify the root `pixi.toml`'s `PYTHONPATH` includes the SODA/torch-mlir python packages and that they are built/installed in this container.
- If PyTorch is missing or the wrong version is installed, create a virtual environment and install a compatible torch wheel from https://pytorch.org.
- **torch-mlir status**: MLIR generation requires torch-mlir, which is currently deferred for aarch64 platforms. The kernel implementations are complete and functional for PyTorch execution. torch-mlir can be built from source when needed for MLIR compilation.

In pixi, torch-mlir can be installed with:

```
pixi run pip install --pre torch-mlir \
  -f https://github.com/llvm/torch-mlir-release/releases/expanded_assets/dev-wheels
```

License & upstream
------------------
The PolyBench-C sources included under `PolyBenchC-4.2.1/` retain their original license and authorship. See `PolyBenchC-4.2.1/LICENSE.txt` and `PolyBenchC-4.2.1/AUTHORS` for details.

