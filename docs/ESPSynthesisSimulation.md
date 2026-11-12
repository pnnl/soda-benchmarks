# Synthesize and simulate an accelerator on ESP (emilia)

Start here for a fresh PyTorch → TOSA → LCS → LLVM → Bambu → ESP full-system
run. Commands below assume a shell on **emilia as Tommaso**. The independent
review checkout is:

```bash
cd /home/tommaso/esp-handoff-20261106/experiments/soda-reuse-review
```

This checkout has its own Git history and build directory. The previous
`soda-reuse-lcs-integration-20261107` checkout retains the measured hardware,
applications, FPGA bitstream and results. You can edit and rebuild the review
checkout without modifying those baseline artifacts.

## 1. Environment and compiler build

The supplied Docker image contains LLVM 19, torch-mlir, SODA and Bambu/PandA.
The installed Python dependencies are mounted read-only. The helpers mount
**this checkout** at `/workspace` and build as your UID, so generated files are
yours. ModelSim and the RISC-V bare-metal toolchain run on the host.

```bash
docker ps                         # Log in again if Docker group access is new.
source examples/esp-validation/emilia/env.sh
mkdir -p build/tutorial
bash examples/esp-validation/emilia/build-plugin.sh > build/tutorial/plugin.log 2>&1
tail -20 build/tutorial/plugin.log
```

The plugin build runs the 67 lit tests and 7 graph-lowering tests. Inspect the
log if the command fails. Rebuild after changing plugin C++ or TableGen files.
For an interactive compiler shell, run `bash examples/esp-validation/emilia/container.sh`.
Inside it, the repository is `/workspace`; exit before the host simulation steps.
Host environment variables are not implicitly forwarded to Docker: put compiler
options inside the command given to `container.sh`, as in the examples below.

The site configuration is in `emilia/env.sh`: `ESP_HANDOFF_ROOT`, `ESP_ROOT`,
`ESP_SOC`, vendor simulation libraries, license server and tool paths. Defaults
use `$HOME/esp-handoff-20261106`. You can export overrides before sourcing it.
The existing compiled Ariane fixture uses DMA64, disabled L2/LLC and 16 MiB RAM.
It already includes ESP drivers, boot ROM and vendor simulation libraries.

## 2. Generate a fixed-graph GEMM accelerator with Bambu

Use a fresh experiment name. This is Tommaso's default fixed-graph task pipeline;
no reusable-operator option is needed.

```bash
bash examples/esp-validation/emilia/container.sh bash -c '
  set -euo pipefail
  python3 -m sb_cli init --base_dir "$PWD/build/tutorial" \
    --output_dir gemm_mini --benchmark_name gemm --dataset MINI \
    --dtype float32 --backend esp --stage llvm \
    --device xc7vx485t-2ffg1761 --clock_period 20
  make -C build/tutorial/experiments/gemm_mini \
    LCS_DMA_WORD_STRIDE=2 LCS_PROFILE=true object cpu-baseline-object
' > build/tutorial/synthesis.log 2>&1
```

`object` produces **both** the RV64 host object and Bambu RTL. The clock target
is 20 ns; HLS does not produce a placed-and-routed FPGA bitstream. The alias
`build/tutorial/experiments/gemm_mini` points at the timestamped experiment.

```bash
generated=$PWD/build/tutorial/experiments/gemm_mini/output/esp/baseline
ls "$generated/bambu_syn/forward_kernel.v" "$generated/06_kernel_riscv.o"
```

| Artifact | Purpose |
| --- | --- |
| `output/01_tosa.mlir` | Imported tensor graph |
| `tasks.mlir`, `dataflow.mlir` | Task creation and scheduled/buffered hardware representation |
| `05_hw_acc.ll`, `architecture.xml` | Hardware LLVM and Bambu interface description |
| `05_kernel_riscv.ll`, `06_kernel_riscv.o` | Host LLVM and RISC-V object |
| `argument_order.txt`, `testdata.h`, `esp-app/` | Generated ABI order, numerical fixture and application |
| `bambu_syn/forward_kernel.v`, `panda_libtech.v`, `*.mem` | Complete generated RTL and initialization files |
| `cpu-baseline/06_kernel_riscv.o` | CPU-only implementation of this fixed graph |

Keep the entire generated directory: copying just `forward_kernel.v` loses
required support RTL and memory initialization files. DMA stride **2** is
required for the emilia DMA64 fixture (one float in the low word of each beat).

## 3. Check the adapter, then run the full ESP system

On the host, the short Icarus test checks the generated accelerator and DMA
adapter against the exported golden values, under immediate and delayed DMA:

```bash
bash examples/esp-validation/generated-lcs/run-adapter.sh \
  "$generated" build/tutorial/adapter > build/tutorial/adapter.log 2>&1
```

Expected: `LCS_DMA64 PASS errors=0` for both stall settings. This test does not
include the CPU or NoC. The following step compiles/links the bare-metal
software, installs the generated RTL into a copy of the compiled ESP fixture,
and runs the actual CPU, NoC, accelerator socket, DMA and memory in ModelSim:

```bash
source examples/esp-validation/emilia/env.sh
bash examples/esp-validation/generated-lcs/run-system.sh \
  "$generated" "$ESP_ROOT" "$ESP_SOC" build/tutorial/system \
  > build/tutorial/system.log 2>&1
```

Use a **fresh output directory for each simulation**. Watch
`build/tutorial/system/full-system/simulation.log` from another terminal.
The script must exit successfully, and the log must contain `TEST PASSED` and
normal program completion. It also records accelerator events and kernel
profiling. ESP ends a successful program with the deliberate VHDL assertion
`Program Completed!`; other assertions or a time limit are not success.
The ELF, software/input hashes and logs remain under the run directory.
Simulation takes minutes, depending on the workload and machine load.

For a CPU-only full-system run of this same small graph, use a fresh directory:

```bash
LCS_SYSTEM_MODE=cpu-baseline bash examples/esp-validation/generated-lcs/run-system.sh \
  "$generated" "$ESP_ROOT" "$ESP_SOC" build/tutorial/cpu-system \
  > build/tutorial/cpu-system.log 2>&1
```

To synthesize another fixed graph, repeat step 2 with a new experiment name and
`--benchmark_name twomm`, for example. GEMM, SYR2K, TRMM, SYMM, SYRK, GESUMMV,
DOITGEN, THREEMM, ATAX and TWOMM pass compiler/native checks. That is not a claim
of full-system validation of all ten kernels. GEMVER fan-out and BICG/MVT
multiple outputs remain unsupported by the fixed-graph scheduler.

## 4. Two differently shaped GEMMs on one reusable engine

Create another experiment, choosing `--benchmark_name gemmreuse`, and build it
with the explicit options below:

```bash
bash examples/esp-validation/emilia/container.sh bash -c '
  set -euo pipefail
  python3 -m sb_cli init --base_dir "$PWD/build/tutorial" \
    --output_dir gemm_reuse --benchmark_name gemmreuse --dataset MINI \
    --dtype float32 --backend esp --stage llvm \
    --device xc7vx485t-2ffg1761 --clock_period 20
  make -C build/tutorial/experiments/gemm_reuse \
    LCS_DMA_WORD_STRIDE=2 LCS_PROFILE=true LCS_REUSABLE_GEMM=true \
    LCS_GEMM_SCHEDULE=lcs object cpu-baseline-object
' > build/tutorial/reuse-synthesis.log 2>&1
```

The same full-system runner accepts this generated directory. Its additional
`reusable-validation.json` checks two launches on instance 0 and 98 outputs.
The second multiplication reads the first one's output in the shared arena.
Dimensions and addresses are descriptor fields; the hardware stays fixed.

The reusable component test has its own fixture and runner; follow the
[reusable GEMM guide](../examples/esp-validation/reusable-gemm/README.md) for
native capture and the two-call/ten-call adapter tests. Do not use the
fixed-graph component fixture from step 3 for a reusable descriptor interface.

`LCS_GEMM_SCHEDULE=sequential` selects the baseline; `lcs` selects overlapping
load/compute/store tasks with ping-pong tile ownership. `LCS_GEMM_TILE_M/N/K`
control hardware tile sizes. Make records schedule and tile choices and
regenerates outputs when they change. Retain separate experiment directories
when comparing implementations so the earlier hardware/results stay available.

## 5. Quick full-system replay of each neural operator

This uses the preserved, already synthesized LCS engines and staged host
objects, rebuilds the runtime/application from this checkout, and creates a
fresh full-system simulation. It is the fastest way to check your installation:

```bash
bash examples/esp-validation/emilia/replay-operator.sh gemm
# The same entry point also accepts conv or add.
```

The printed JSON must show success: GEMM has 2 calls/6 outputs, convolution
3 calls/60 outputs, and addition 2 calls/143 outputs. Each run gets a new
`build/replay-KIND.*` directory with software and simulator logs. This is a
replay of saved hardware; use the next section to regenerate it from MLIR.

## 6. Synthesize fresh reusable convolution, addition and GEMM

The hardware generator takes **no model input**. Its fixed resources and schedule
are independent of the model's dimensions and tensor addresses. In a fresh
output directory, generate all engines and compile the three small operator apps:

```bash
bash examples/esp-validation/emilia/container.sh bash -c '
  set -euo pipefail
  NN_CONV_SCHEDULE=lcs NN_GEMM_SCHEDULE=lcs \
    bash scripts/reusable_nn/generate_hardware.sh build/tutorial-nn/hardware
  python3 scripts/reusable_nn/combine_rtl.py build/tutorial-nn/hardware
  python3 examples/esp-validation/reusable-nn/export_operators.py build/tutorial-nn/operators
  for kind in conv add gemm; do
    model=build/tutorial-nn/operators/$kind
    bash scripts/tosa_to_reusable_nn.sh "$model/01_tosa.mlir" "$model/compiled"
    python3 models/pytorch/resnet18/run_native.py "$model"
    python3 examples/esp-validation/reusable-nn/stage_app.py "$model" "$model/esp-app"
  done
' > build/tutorial/nn-synthesis.log 2>&1
```

Then run all three on the host using the newly generated hardware and apps:

```bash
source examples/esp-validation/emilia/env.sh
for kind in conv add gemm; do
  model=build/tutorial-nn/operators/$kind
  bash examples/esp-validation/reusable-nn/build-software.sh \
    "$model/esp-app" "$ESP_ROOT" "$ESP_SOC"
  bash examples/esp-validation/reusable-nn/run-system.sh \
    build/tutorial-nn/hardware "$model/esp-app" "$ESP_SOC" "$model/system"
done
```

Each `system/full-system/validation.json` must pass. Keep the results and
`validation-inputs.sha256`: FPGA packaging checks them against the exact RTL
and software inputs. See the [neural-operator guide](../examples/esp-validation/reusable-nn/README.md)
for randomized DMA-stall cosimulation, the wider-channel convolution fixture,
and complete ResNet-18 export/compilation.

The same three engines run ResNet-18 through 30 sequential host offloads
(20 convolutions, 9 additions, 1 GEMM), with BatchNorm folding and supported
ReLU fusion. The CPU retains layout conversion and pooling. The complete model
needs about 98 MiB of shared memory, so use individual operator full-system
RTL tests with the 16 MiB simulation fixture and the FPGA for full-model execution.

## 7. Optional FPGA implementation after simulation

Bambu HLS and full-system simulation above do not require programming a board.
After all three fresh operator tests pass, package and implement their exact RTL:

```bash
python3 examples/esp-validation/reusable-nn/stage_hardware.py \
  build/tutorial-nn/hardware build/tutorial-nn/esp-package
python3 examples/esp-validation/generated-lcs/prepare-soc.py \
  build/tutorial-nn/esp-package "$ESP_ROOT" "$ESP_SOC" build/tutorial-nn/esp
bash examples/esp-validation/reusable-nn/build-fpga.sh \
  build/tutorial-nn/operators system build/tutorial-nn/esp/socs/vc707 build/tutorial-nn/fpga
```

Vivado 2023.2 must be on PATH for this optional step. The
[neural-operator guide](../examples/esp-validation/reusable-nn/README.md) covers
physical optimization, full-model software with the board's 1 GiB map, UART
capture and JTAG loading. Use your board connection and reserved target.
[Integration results](ESPReuseLCSIntegration.md) identify the preserved validated
bitstream and measurements; no new board run is needed to follow steps 1–6.

## Troubleshooting

- Docker permission denied: start a new login so membership in `docker` applies.
- Plugin/pass missing: rebuild with this checkout's `build-plugin.sh`; do not
  point at a plugin from the older compiler checkout.
- ModelSim library/license error: source `emilia/env.sh` in the host shell and
  check `ESP_MODELSIM_INI`, `ESP_MODELSIM_WORK` and license availability.
- Missing `.mem` or unknown Bambu module: keep the whole generated RTL directory.
- Existing output/library error: choose a fresh simulation directory.
- Simulation time limit: inspect the CPU/RTL logs first; it is not a pass. The
  helpers default to a 2000 ms simulated-time limit and bounded DMA timeouts.
- Memory-size mismatch: keep `NN_DRAM_BYTES=16777216` for operator simulation;
  the full-model board application uses `1073741824` instead.
