# A SiliconCompiler builder for `sb-cli`

`sb-cli init` scaffolds one experiment directory per build. The `--builder`
option chooses the build script it is driven by:

- `make` (the default) writes the `Makefile` that chains the `scripts/`
  wrappers around each tool and finishes by handing an
  OpenROAD-flow-scripts `config.mk` to `make`.
- `siliconcompiler` writes `sc_flow.py` **beside** that Makefile, which drives
  the same tools — soda-opt, Bambu, Yosys, OpenROAD, KLayout — as a single
  [SiliconCompiler](https://docs.siliconcompiler.com) job.

The builder is additive. An experiment scaffolded with `--builder
siliconcompiler` still has every file the default builder writes, byte for
byte, and still builds with `make`; the two backends are comparable against
each other because selecting one does not change what the other builds.

## Selecting it

```bash
pixi run sb-cli init \
    --benchmark_name PolyBenchPyTorch.linear_algebra.blas.gemm \
    --dataset MINI \
    --dtype float32 \
    --device nangate45 \
    --clock_period 5 \
    --flow optimized \
    --stage gds \
    --builder siliconcompiler \
    --output_dir gemm_sc

cd benches/experiments/gemm_sc
pixi run -e sc ./sc_flow.py    # or: make, which is unchanged
```

Scaffolding is a default-environment job — `sb-cli` only *writes* `sc_flow.py`
and never imports SiliconCompiler — but **running** the generated script needs
the `sc` environment, hence the `-e sc`.

`sb-cli fork` carries `sc_flow.py` along, so a fork of a SiliconCompiler
experiment stays buildable by it.

## What it needs

### The `sc` pixi environment

SiliconCompiler is deliberately not in the default environment: nothing in this
repository imports it, so the default install stays small and solves without a
git clone. It lives in the `sc` feature of `pixi.toml`, which the `sc`
environment is built from:

```toml
[feature.sc.pypi-dependencies]
siliconcompiler = { git = "...", rev = "3f33a3c5" }

[environments]
sc = ["sc"]
```

Prefix any command that imports the package with `-e sc`:

```bash
pixi install -e sc                       # solve it once
pixi run -e sc python -c "import siliconcompiler.flows.sodaflow"
pixi run -e sc ./sc_flow.py
```

The `sc` environment is a superset of the default one — the editable install of
this repository and the same `PYTHONPATH` are in both — so `sb-cli` works there
too if running everything under `-e sc` is simpler.

### A build that carries the SODA support

`siliconcompiler.flows.sodaflow`, `siliconcompiler.tools.soda` and
`siliconcompiler.tools.mlir` are what the generated script imports, and no
released wheel ships them yet — a plain `siliconcompiler` from PyPI gets a
package whose import block fails immediately. That is why the dependency is
pinned to a git revision rather than a version; move it to a version constraint
once a release carries them. `pixi run -e sc python -c "import
siliconcompiler.flows.sodaflow"` is the check either way.

### The tools on `PATH`

The tools it drives are the ones the Makefile backend drives, and it finds them
on `PATH` rather than in the pixi environment: `mlir-opt`/`mlir-translate`,
`soda-opt` and `bambu` for every stage, plus `yosys`, `openroad`, `sta` and
`klayout` for `--stage gds`. The PDK data comes from lambdapdk, which
SiliconCompiler installs as a Python package into the `sc` environment, so there
is no OpenROAD-flow-scripts checkout to point at.

Bambu has to be built against clang 15 or newer: the LLVM IR `mlir-translate`
emits carries opaque pointers, which an older Bambu front end rejects. The
generated script pins `--compiler=I386_CLANG16`, the same compiler
`scripts/ll_to_verilog.sh` pins, and `BAMBU_COMPILER` at the top of the script
is where to change it.

### Tests

The one test that executes a generated `sc_flow.py` is marked `sc`, so it is
deselected in the default environment and run in the other:

```bash
pixi run test              # everything but the sc-marked tests
pixi run -e sc test-sc     # only those
pixi run -e sc test-all    # the whole suite in one environment
```

Every other test of the builder — that the Makefile is untouched, that no
placeholder survives, that `fork` carries the script — only reads the generated
text, so it stays in the default environment.

## What the template substitutes

`sb_cli/templates/sc_flow.py.tmpl` is a `string.Template` like
`Makefile.tmpl`, and every placeholder it uses is a key `sb_cli.init.scaffold()`
already builds for the other templates:

| Placeholder | Source |
|---|---|
| `$experiment_name` | the logical experiment name — `--output_dir` or the derived `<benchmark>-<dataset>-<dtype>[-NNN]`. It is the symlink name rather than the timestamped directory name, and the SiliconCompiler job name. |
| `$benchmark_name_repr` | `repr(ExperimentConfig.benchmark_name)`, so a Python file gets `None` and not `"None"`. `$benchmark_name` is the human-readable form the other templates use. |
| `$dataset`, `$dtype`, `$device`, `$clock_period`, `$memory_policy`, `$flow`, `$backend`, `$stage`, `$instrumentation` | the `ExperimentConfig` fields of the same name |
| `$created_at` | when `sb-cli init` ran |

Rendering it is substitution and nothing else: there is no generated block of
code to keep in step, which is why offering the builder costs `sb_cli` one
extra `render()` call.

## Stage by stage

`--stage` selects how far to build, the same four values either builder takes:

| `--stage` | Makefile backend | SiliconCompiler builder |
|---|---|---|
| `llvm` | `output/05_llvm_<flow>.ll` | `ElaborationFlow(frontend=...)` stopped at the `link` node, whose output is that IR |
| `verilog` | `output/bambu/<flow>/06_verilog.v` | the same flow run to its exit node, which is Bambu |
| `simulation` | `output/bambu/<flow>/07_results.txt` | as above, plus `set_bambu_simulate(True)`; the C testbench soda-opt emits reaches Bambu over the flow, so nothing else is needed |
| `gds` | ORFS `6_final.gds`, via the generated `config.mk` | `ASICFlow(frontend=...)`: synthesis, floorplan, place, CTS, route and GDSII export are SiliconCompiler's own steps |

`--flow` selects how soda-opt lowers the outlined kernel, and each strategy is
a flow handed to the one above as its `frontend`:

| `--flow` | Front end |
|---|---|
| `baseline` | `SODABaselineElaborationFlow` |
| `optimized` | `SODAOptimizedElaborationFlow` |
| `transformed` | `SODATransformedElaborationFlow`, given the experiment's `transform.mlir` as its schedule |

## Instrumentation IPs

`sb-cli init --instrumentation <recipe>` copies the recipe's `IPs/` directory
into the experiment verbatim, and `sb-cli fork` carries it along, so the
directory already is the recipe's file list and there is nothing for `sb-cli`
to tell the script that the directory does not say. `configure_ip()` reads it
and sets the high-level synthesis task's parameters:

| `IPs/` holds | Task setting | `ll_to_verilog.sh` equivalent |
|---|---|---|
| `module_lib.xml` | `set_bambu_technologyfile()` | `IP_MODULE_LIB` |
| `constraints_STD.xml` | `set_bambu_constraintsfile()` | `IP_CONSTRAINTS` |
| `*.c` | `add_bambu_cnoparse()` | `IP_C_EXCLUDE` |
| `*.v` | `add_bambu_fileinputdata()` | `IP_VERILOG_INPUTS` |
| any of the above | `set_bambu_componentslibrary(True)` | `--generate-components-library` |

`module_lib.h` is read by the C stub rather than by Bambu, and neither backend
passes it. The `none` recipe copies no `IPs/`, and the same code then sets
nothing.

Going through the task's parameters rather than raw command line options means
SiliconCompiler treats these as files: they are hashed into the node's cache
key, so changing an IP re-runs high-level synthesis, and they are copied for a
remote run.

The table maps content, not command lines. The two backends hand Bambu the same
two XMLs in opposite positional slots — `ll_to_verilog.sh` emits
`module_lib.xml constraints_STD.xml ... input.ll`, SiliconCompiler emits
`<source>.ll constraints_STD.xml module_lib.xml` — and Bambu's documented
signature is `bambu <source> [constraints] [technology]`, which upstream's own
order already contradicts. That implies Bambu classifies them by content rather
than by position, but nobody has built an instrumented experiment both ways to
confirm it, so treat an instrumented build as the thing to check first if the
generated RTL differs.

## What it does not carry

- **`--backend`.** It has one value, `bambu`, which is what the script assumes.
  SiliconCompiler still has Bambu consume the LLVM IR, so `--builder` is an
  axis of its own rather than another `--backend` value.
- **Devices.** `TARGETS` in the script maps `nangate45` onto
  `freepdk45_demo` and `asap7` onto `asap7_demo`, the two platforms the ORFS
  path has `config.mk` files for. A device with no entry is an error rather
  than a silent fallback.
- **Corners.** The device Bambu estimates against comes from the target's main
  library — which is what keeps it in step with the cells synthesis maps to —
  so a corner suffix `asap7_demo` does not carry is not honoured: it pins
  `asap7-WC`. The script says so on stdout instead of quietly reporting another
  corner's numbers.
- **`--dtype`.** It is baked into `torchscript.py`, and reaches the build as the
  types in `output/01_tosa.mlir`.
- **Bambu's graphviz dumps.** `-v3` is emitted (the resource summary is only
  printed at that verbosity) but `--print-dot` is not; SiliconCompiler exposes
  it as `set_bambu_printdot()`.

## Replacing the generated Makefile

The generated `Makefile` chains a shell script around each tool and ends by
handing an OpenROAD-flow-scripts `config.mk` to `make`. The SiliconCompiler
builder replaces the whole of it, one artifact at a time:

| Generated Makefile | SiliconCompiler builder |
|---|---|
| `$(ODIR)/01_tosa.mlir` from `torchscript.py` | `export_model()`, unchanged. Its output is the design's source fileset. |
| `tosa_to_llvm.mk`: `02_linalg.mlir` via `tosa_to_linalg.sh` | the `tosa2linalg` and `bufferize` nodes |
| `tosa_to_llvm.mk`: `03_llvm.mlir`, `04_llvm.ll` via `linalg_to_llvm.sh` and `llvm_to_ll.sh` | not on the hardware path either way — that is the reference CPU lowering, which SiliconCompiler has as `LinalgToLLVMTask` for anyone who wants it |
| `soda_to_llvm.mk`: `04_llvm_<flow>.mlir` via `soda_to_llvm_<flow>.sh` | the `soda` node, which is whichever of the three front-end flows was selected |
| `$(ODIR)/04_transform_sched.mlir`, copied from `transform.mlir` | `TransformedTask.set_soda_schedule()` |
| `mv forward_kernel_testbench.c $(ODIR)/` | an output of the `soda` node, staged into the Bambu node by the flow, so no path is agreed on twice |
| `mlir-translate --mlir-to-llvmir` | the `translate` node |
| `sed -E '/llvm\.stacksave\.p0\|llvm\.stackrestore\.p0/d'` | `TranslateTask.add_mlir_stripintrinsics()`, whose default is those two intrinsics |
| `opt -S -o $@ -` after that `sed` | nothing. The stripped IR goes straight to `llvm-link`; the round trip through `opt` is the one step of this pipeline the builder does not reproduce. |
| `link_memref_copy.sh` | the `runtime` and `link` nodes. `link` merges only when the kernel really references `memrefCopy` without defining it, so a kernel with no use for the helper does not carry a dead function into synthesis. |
| `llvm_to_verilog.mk`, `ll_to_verilog.sh`, `BAMBU_CLOCK_PERIOD`, `BAMBU_DEVICE` | the `convert` node. The period comes from the SDC and the device from the target's main library, so neither is stated twice. |
| `BAMBU_MEMPOLICY` | `set_bambu_memorypolicy()` |
| `BAMBU_RUN_SIMULATION=true` (the `07_results.txt` rules) | `set_bambu_simulate()`, with `set_bambu_simulator()` and `set_bambu_verilatorparallel()` |
| `BAMBU_IP_INTEGRATION`, `IP_*` | the task parameters in [Instrumentation IPs](#instrumentation-ips) |
| `BAMBU_TOP_FNAME ?= forward_kernel` | the design's topmodule, which everything downstream already reads |
| `patch_openroad_synt.sh`, `<platform>_config.mk`, `<platform>_constraints.sdc`, `synthesize_Synthesis_kernelname.sh` | nothing. The target supplies the technology and the design supplies the SDC; there is no generated backend script to patch. |
| `verilog_to_gds.mk`, `verilog_to_gds.sh`, `GDS_PLATFORM` | the rest of `ASICFlow` |

## Why bother

- **No generated backend.** `patch_openroad_synt.sh`, the per-platform
  `config.mk`, the per-platform `.sdc` and `synthesize_Synthesis_<kernel>.sh`
  all go away. The target supplies the technology and the design supplies the
  constraints.
- **The clock period is stated once.** Today it appears in the Makefile, in the
  Bambu invocation and again in the platform `.sdc`, and nothing checks that the
  three agree. The script writes one `constraints.sdc`, in the library's own
  time unit — 5 for nangate45, 5000 for asap7, exactly the difference between
  the two platform SDC files — and Bambu, synthesis and place-and-route all
  read it.
- **Intermediate files stop being shared paths.** Each node writes into its own
  `outputs/` and the next node's `inputs/` are staged from it, so a stage cannot
  read a stale artifact from an earlier run.
- **The strategies stop being `TARGET=` string surgery** and become an argument.
- **The build is recorded.** One manifest per job, holding the tool versions,
  the metrics and the parameters every task ran with — which is what makes a
  baseline-against-optimized comparison reproducible rather than remembered.

## Further reading

- The SODA front-end flows, and the `frontend` argument every downstream flow
  takes:
  [predefined flows](https://docs.siliconcompiler.com/en/latest/reference_manual/predef_modules/flows.html).
- The SODA tutorial, which walks the same path from a model to GDSII. It is
  published as `user_guide/tutorials/soda` alongside the SODA support described
  above, so it is not on the documentation site until that ships; until then it
  is `docs/user_guide/tutorials/soda.rst` in the SiliconCompiler source.
