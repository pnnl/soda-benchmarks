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

## `polybench_mini_TS/`

The upstream DSE output these runs came from, kept for provenance. It is not
read at run time: `sb-cli init --default_ts alo_soda` reads the merged copies
shipped in `sb_cli/schedules/alo_soda/MINI/`. See
[`sb_cli/schedules/README.md`](../../sb_cli/schedules/README.md).
