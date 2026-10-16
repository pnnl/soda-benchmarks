# Running many experiments in parallel

Scaffolds the PolyBench MINI HLS matrix and drives it with GNU `parallel`.

| Axis | Values | Count |
|---|---|---|
| Kernel | atax bicg doitgen gemm gemver gesummv mvt symm syr2k syrk threemm trmm twomm | 13 |
| Device | `nangate45`, `xcu280-2Lfsvh2892-VVD` | 2 |
| Instrumentation | none, assert, change-location, change-location-exposed-ports, hw-counters | 5 |
| Transform schedule | `simple`, `alo_soda` | 2 |

260 simulation jobs, and 130 synthesis jobs (nangate45 only — see below).

## Usage

```bash
export BASE_DIR=$HOME/soda-runs/polybench-mini

./setup.sh              # scaffold + write the job files
./run-simulation.sh     # make simulation, in parallel
./run-synthesis.sh      # make gds, in parallel -- AFTER run-simulation.sh
```

`BASE_DIR` must be the same for all three: the runners read their job files from
it, and `parallel --resume` keys off the joblog stored beside them.

| Variable | Default | Meaning |
|---|---|---|
| `BASE_DIR` | `$HOME/soda-runs/polybench-mini` | Where experiments and job files go |
| `PIXI_CMD` | `pixi run --manifest-path <repo>/pixi.toml` | How each job enters the environment |
| `JOBS` | 4 (simulation), 2 (synthesis) | Concurrent jobs |
| `RESUME` | 1 | Set to 0 to re-run jobs the joblog already records |

`setup.sh` also exposes `DATASET`, `DTYPE`, `CLOCK_PERIOD`, `FLOW`, `BACKEND` and
`STAGE`, and the four matrix arrays are declared at the top of the file — edit
them there to run a slice.

## Run simulation before synthesis

Both `make simulation` and `make gds` reach Bambu through
`scripts/ll_to_verilog.sh`, and the simulation run already copies
`forward_kernel.v` to `06_verilog.v` and writes the OpenROAD scripts for the
device. So simulation-then-synthesis invokes Bambu **once**; the reverse order
invokes it twice *and* leaves `06_verilog.v` newer than the GDS, forcing a full
re-synthesis.

For the same reason the two scripts must not run at the same time against one
experiment: `06_verilog.v` and `07_results.txt` are separate Bambu invocations
writing into the same `output/bambu/transformed/` tree.

## Synthesis is nangate45 only

`xcu280-2Lfsvh2892-VVD` falls through the device `case` in
`scripts/ll_to_verilog.sh`, so no OpenROAD `config.mk` or
`synthesize_Synthesis_forward_kernel.sh` is ever generated for it, and
`scripts/to_copy/` carries no Xilinx equivalent. There is no Vivado in this
container either. That device therefore stops at simulation, and `setup.sh`
emits no synthesis jobs for it.

## Layout and resumability

```
$BASE_DIR/
  experiments/<kernel>__<device>__<instrumentation>__<schedule>/   # symlink
  jobs-simulation.txt    jobs-simulation.joblog
  jobs-synthesis.txt     jobs-synthesis.joblog
  logs/init-<name>.log
```

Each job writes `build-simulation.log` / `build-gds.log` inside its own
experiment directory.

`setup.sh` is idempotent: re-running skips experiments that already exist but
still re-emits their job lines, so the job files stay complete. `parallel
--resume` then skips whatever the joblog already records, which is what makes a
260-job sweep restartable. It keys off the **input line number**, so do not
reorder or hand-edit a job file between runs.

`sb-cli init` runs sequentially on purpose — it read-modify-writes
`experiments/registry.py` and creates a symlink, so parallelising it would race
both.

## Collecting results and building the report

`results/` holds one directory per campaign (e.g. `mini-none-simple`,
`mini-none-alo_soda`) and an `index.html` report comparing them. The scripts
live in `results/tools/`:

```bash
cd results
tools/collect.sh $HOME/soda-runs/polybench-mini-none-simple mini-none-simple
tools/collect.sh $HOME/soda-runs/polybench-mini-none        mini-none-alo_soda
pixi run --manifest-path ../../../pixi.toml python tools/parse_logs.py .    # -> data.json
pixi run --manifest-path ../../../pixi.toml python tools/build_report.py .  # -> index.html
```

| Script | Does |
|---|---|
| `collect.sh <campaign> <dest> [--with-layout]` | Copies the joblogs, each experiment's `build-*.log`, the OpenROAD `reports/.../base` (`.rpt`, `synth_stat.txt`, layout PNGs) and the per-stage ORFS metrics JSON. `--with-layout` also copies `.odb`/`.gds`/`6_final.v`, which the report does not need. |
| `parse_logs.py <results> [config ...]` | Streams every log into `data.json`. Defaults to every directory holding a `jobs-*.joblog`. Stdlib only. |
| `build_report.py <results>` | Inlines `data.json` into `tools/report_template.html` and writes `index.html`. |

`index.html` opens straight from disk (the data is inlined; Chart.js comes from
cdnjs). Layout images are linked relatively, so keep it in `results/`.

Each campaign is collected as:

```
results/<config>/
  jobs-simulation.joblog  jobs-synthesis.joblog  jobs-*.txt
  logs/<kernel>/
    build-simulation.log   # Bambu HLS + Verilator: cycles, FFs, est. area/Fmax, FU summary
    build-gds.log          # Yosys + OpenROAD: per-step elapsed time and peak memory, errors
    base/                  # 6_finish.rpt (slack, power), synth_stat.txt (cell area), *.webp.png
    metrics/<stage>.json   # ORFS metrics, 2_1_floorplan.json ... 6_report.json
```

### Where each number comes from

| Report section | Source |
|---|---|
| Job status | `build-*.log`: a sim run is ok when it prints `Total cycles`; a GDS run fails on `[ERROR`, `Error:` or a `make *** Error`, and the step that failed is taken from `Command exited with non-zero status`. |
| Job time / timeline | `jobs-*.joblog` (`Starttime`, `JobRuntime`). |
| Cycles, flip-flops, Bambu resources | `build-simulation.log`, section for `forward_kernel`. |
| Cell area | `base/synth_stat.txt` (`Chip area for module`). |
| Slack, power | `base/6_finish.rpt`. |
| ORFS metrics (instance count, explorer) | `metrics/<stage>.json`. |
| Layout die size and ratio | `design__die__area`, `design__instance__area__stdcell` and `design__instance__utilization` from `6_report.json`, or from the last stage reached. The side length is √area, assuming a square die. |
| Flow step time and memory | the `Elapsed time: … Peak memory:` line after each ORFS step in `build-gds.log`. Unlike the summary table ORFS prints at the end, this also covers runs that failed. |

**Do not rely on the joblog alone to tell whether a job worked.** In
`mini-none-alo_soda`, syrk's `make gds` failed in global routing
(`GRT-0116`, after 197 h and 92 GB) but its joblog row says `Exitval 0`. The
report reads status from the logs.

### ORFS metrics JSON

OpenROAD writes `logs/nangate45/forward_kernel/base/<stage>.json` after each
stage. Every stage uses one schema: the keys are prefixed with the stage name
(`floorplan__`, `globalplace__`, `placeopt__`, `detailedplace__`, `cts__`,
`globalroute__`, `detailedroute__`, `finish__`) and cover instance count/area,
utilization, die/core area, setup/hold slack and violation counts, power,
wirelength, DRC errors per routing iteration, and warning/error counts. For
example, `finish__design__instance__count` in `6_report.json` is the number of
standard-cell instances without fill.

They are the most convenient OpenROAD source. One parser covers every stage,
and a run that fails later still keeps the stages it finished: syrk/alo_soda
stops at `5_1_grt.json`, and the report falls back to its CTS instance count.
The report's metric explorer can plot any numeric key at any stage for every
configuration, and trace one metric across the stages for one kernel.

## `polybench_mini_TS/`

The upstream DSE output these runs came from, kept for provenance. It is not
read at run time: `sb-cli init --default_ts alo_soda` reads the merged copies
shipped in `sb_cli/schedules/alo_soda/MINI/`. See
[`sb_cli/schedules/README.md`](../../sb_cli/schedules/README.md).
