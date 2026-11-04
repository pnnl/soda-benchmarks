# Generated PolyBench accelerators on VC707

This campaign generates a separate fixed-shape accelerator and matching host
application for each of GEMM, SYR2K, TRMM, SYMM, SYRK, GESUMMV and DOITGEN. It
uses MINI float32 inputs and the unchanged absolute validation threshold
`1e-3`. The hardware is generated through SODA and Bambu; it is not the older
FFN accelerator executing these kernels through batch-matmul calls.

Completed on 4 November 2026: all seven generated host/accelerator pairs passed
full-system RTL simulation, all seven CPU baselines passed, and all seven
VC707 bitstreams were produced with nonnegative reported setup/hold slack.
No physical-board run was performed in this campaign.

The final campaign lives at `local-results/vc707-20261104-buffered` in the validation
checkout. `status.json` records progress; `results.csv` and `results.json`
collect completed runs. Board testing is a subsequent, manual step.
The `fpga` status in those files means successful bitstream generation and
report checks; it does not mean an application has run on the physical board.

The starting point is PR #17 at `e5d2515`. Generation used the local fixes
through `77fb9c4`; the buffered adapter is commit `6a3b19a`, and the campaign
automation is `20418d8`. These are local validation-branch commits. The ESP
source is based on `900ae43`, with the existing compiled SoC fixture described
below. `environment.json` records full hashes, tool versions and local paths.

The first campaign, `local-results/vc707-20261103`, generated all seven hardware
and software pairs afresh and exposed an early-write deadlock in GESUMMV.
The final set reuses those unchanged generated artifacts and rebuilds the
integrations with bounded write buffering in the adapter. The failure, VCD,
and correction are described in
[ESPNativeDMAWriteBuffer.md](ESPNativeDMAWriteBuffer.md).
Some final-run directories link back to that preserved initial campaign;
retain both directories. The completed `deployment-and-evidence.tar.gz`
contains actual deployment files and reports for transfer, without relying
on those links. It does not bundle the ESP checkout or EDA tools.

## Generation and integration

For each benchmark, the generation command is:

```sh
sb-cli init --base_dir "$CAMPAIGN/generated" --output_dir "$kernel" \
  --benchmark_name "$kernel" --dataset MINI --dtype float32 \
  --flow transformed --backend esp --stage object \
  --device xc7vx485t-2ffg1761 --clock_period 20
make -C "$CAMPAIGN/generated/experiments/$kernel" \
  LCS_PROFILE=true LCS_DMA_WORD_STRIDE=2 object cpu-baseline-object verilog
```

Run generation in the configured SODA container. Its `/workspace` mount is
this validation checkout; use a campaign path inside that mount so the host
can subsequently read the outputs. Repeat for the seven names above.

The host and accelerator pipelines consume the same outlined kernel. The
host uses the runtime to allocate shared storage, copy inputs, program the
ESP socket, launch/wait and copy outputs. The hardware pipeline generates
load/compute/store LLVM and architecture XML for Bambu. The CPU baseline
compiles the same outlined tiled/fused kernel without offloading it.

`LCS_DMA_WORD_STRIDE=2` allocates one float in the low 32 bits of each 64-bit
slot. Matching host and hardware address calculations are generated together.
This supports the existing adapter's aligned writes, at the cost of doubling
tensor storage and DMA traffic. It is a compatibility layout, not a bandwidth
optimization. Do not mix dense and padded host/RTL artifacts.

The scripts under `examples/esp-validation/generated-lcs/` perform these steps:

1. `prepare-soc.py` creates an isolated VC707 ESP integration from the validated
   SoC fixture, copies Bambu RTL and memory initialization files without edits,
   and adds the existing 32-to-64-bit DMA adapter. It records input checksums.
2. `run-adapter.sh` checks numerical results, input/padding preservation,
   repeated invocations and backpressure at the adapter boundary.
3. `run-system.sh` runs the original generated host object and accelerator
   together with the RISC-V processor, ESP socket, NoC and memory tile. A
   second run executes the matching CPU baseline. A compiled SoC fixture is
   reused, with a fresh accelerator library and application per run.
4. `build-fpga.sh` permits FPGA synthesis only after the offload simulation
   passes. Vivado synthesizes, places, routes and writes a bitstream, then
   checks setup/hold slack and the system clocks.
5. `collect-results.py --require-complete` checks the full campaign, verifies
   that the installed generated RTL remains unchanged, gathers measurements
   and checksums all deployment files, including the CPU executable.

`run-campaign.py` coordinates the steps, with separate bounded pools for
simulation and FPGA implementation. Use a fresh campaign directory for a new
run. The generated inputs must already exist under
`$CAMPAIGN/generated/experiments/<kernel>/output/lcs`.

From the host, after setting the existing ModelSim/license and RISC-V
environment (`LD_LLD=ld.lld-16` on this machine):

```sh
python3 examples/esp-validation/generated-lcs/run-campaign.py \
  "$CAMPAIGN" "$ESP_ROOT" "$COMPILED_SOC" --sim-jobs 2 --fpga-jobs 2
python3 examples/esp-validation/generated-lcs/collect-results.py \
  "$CAMPAIGN" --require-complete
```

`ESP_MODELSIM_WORK` and `ESP_MODELSIM_INI` select the existing compiled SoC
and vendor libraries. `COMPILED_SOC` is the matching SoC fixture directory.
These dependencies, paths and tool versions are recorded in the campaign's
`environment.json` and driver logs. The Vivado build uses the installed license
environment and targets `xc7vx485tffg1761-2`.

`--cpu-results <previous-campaign>` can reuse passing CPU baselines after
checking identical host objects, input data and the same simulation fixture.
The baseline must not have invoked the accelerator. This was used for the
adapter-only rebuild; it does not reuse offload simulation results.

The SoC is the existing 2-by-3 validation configuration: one Ariane processor,
one I/O tile, one memory tile connected to DDR3, one accelerator tile and two
empty tiles. ESP L2/LLC are disabled. DMA is noncoherent. CPU/accelerator clocks
are 50 MHz; the memory/NoC domain is 100 MHz. The accelerator retains the
fixture's `sld,sqr_bambu` identity (`0x07a`); its implementation is replaced for
each kernel. This automates integration into that known SoC, not arbitrary
tile placement or hardware interface discovery.

## Evidence and performance interpretation

All seven generated host/accelerator pairs and their CPU baselines passed
full-system RTL simulation. These measurements use the buffered adapter:

| Kernel | CPU kernel cycles | Full offload cycles | Hardware cycles | Max error (e-9) |
|---|---:|---:|---:|---:|
| GEMM | 403,082 | 1,064,746 | 58,462 | 1,907 |
| SYR2K | 926,898 | 1,289,580 | 54,445 | 2,861 |
| TRMM | 308,498 | 933,552 | 35,616 | 1,907 |
| SYMM | 312,362 | 1,058,595 | 49,518 | 15,259 |
| SYRK | 482,906 | 1,157,615 | 31,136 | 1,907 |
| GESUMMV | 53,514 | 819,348 | 11,478 | 3,815 |
| DOITGEN | 293,780 | 1,285,305 | 144,811 | 238 |

The absolute error threshold is `1,000,000 e-9` in every case. Each accelerator
run and its CPU baseline report the same maximum error. These are
separate generated accelerators, not seven workloads run on the same FFN IP.

Each kernel directory contains:

| Path | Contents |
|---|---|
| `adapter/` | Component RTL numerical/backpressure test |
| `system-esp/full-system/simulation.log` | RISC-V host plus accelerator, numerical check and profiling |
| `system-cpu/full-system/simulation.log` | Matching CPU baseline and profiling |
| `esp/integration.json` | Source paths, socket identity and installed RTL checksums |
| `fpga/reports/` | Timing, clocks, utilization, hierarchy and DRC reports |
| `deploy/` | Matching `.bit`, offload `.exe`, CPU `.exe`, boot ROM and checksums |

Compare the `kernel` region (7) between offload and CPU runs. The offload
includes allocation/setup, packing, runtime launch/wait, unpacking and cleanup.
`accel` (2) measures launch/wait only; the adapter also reports the hardware
execution interval independently. All measurements here are single cold
invocations. Device/cache probing, including UART messages, is inside the
allocation region. These numbers are not warm invocation throughput.

All seven cold offloads take longer than their CPU baseline end to end,
despite the shorter hardware execution intervals. The measurements identify
allocation/probing and copies as the dominant costs for these MINI inputs;
they establish correctness, not an application speedup.

The full-system simulation uses ESP's memory model, not the implemented DDR3
controller's physical timing. Its cycle counts establish a reproducible
simulation comparison; measure the FPGA separately before claiming board
performance. The adapter stress-test durations are not performance numbers.

ESP signals normal application completion with `** Failure: Program Completed!`.
The scripts require that specific completion signal and `TEST PASSED`, while
rejecting other simulator failures, errors and timeouts.

## FPGA implementation results

All images target the VC707's `xc7vx485tffg1761-2`, with a 50 MHz
processor/accelerator and 100 MHz NoC/memory domain. Slack is the worst reported
value across the constrained SoC paths. Resource counts below cover the
generated accelerator plus its DMA adapter, excluding the socket and other
SoC tiles; complete SoC utilization is retained in `results.json` and the
Vivado reports.

| Kernel | Setup slack (ns) | Hold slack (ns) | LUTs | FFs | DSPs | BRAM36 / BRAM18 |
|---|---:|---:|---:|---:|---:|---:|
| GEMM | 0.575 | 0.054 | 5,675 | 4,005 | 4 | 1 / 20 |
| SYR2K | 0.785 | 0.050 | 8,306 | 5,712 | 6 | 2 / 32 |
| TRMM | 0.094 | 0.051 | 4,517 | 3,496 | 4 | 1 / 16 |
| SYMM | 0.888 | 0.054 | 6,609 | 4,639 | 4 | 1 / 24 |
| SYRK | 0.423 | 0.052 | 5,532 | 3,967 | 4 | 1 / 20 |
| GESUMMV | 0.552 | 0.053 | 5,517 | 3,793 | 6 | 2 / 28 |
| DOITGEN | 0.396 | 0.052 | 4,666 | 3,783 | 2 | 0 / 0 |

Vivado's reported setup/hold slack covers constrained paths. The reused ESP
fixture retains two reset-controlled latch endpoints in the memory tile's
`noc2aximst` proxies without clock constraints, plus peripheral I/O delay
warnings. These are listed in `check_timing.rpt`; positive slack is not a
claim that every SoC path is constrained. The build separately checks that
the accelerator's sequential logic is on the intended 50 MHz clock. DRC
warnings and advisory messages are retained in `drc.rpt` and `results.json`.

## Manual FPGA test

Open and record the UART on the board host as usual. These commands do not
open or configure the UART. From the validation checkout:

```sh
export CAMPAIGN="$PWD/local-results/vc707-20261104-buffered"
export XILINX_VIVADO=/opt/xilinx/Vivado/2023.2
export PATH="$XILINX_VIVADO/bin:/opt/riscv/bin:$PATH"
export FPGA_HOST=espdev    # replace if the JTAG server is elsewhere
kernel=gemm
soc="$CAMPAIGN/$kernel/esp/socs/vc707"
deploy="$CAMPAIGN/$kernel/deploy"

(cd "$deploy" && sha256sum -c SHA256SUMS)
make -C "$soc" fpga-program FPGA_HOST="$FPGA_HOST"
make -C "$soc" fpga-run TEST_PROGRAM="$deploy/$kernel.exe"
```

Use your normal Vivado license environment. Supply `ESPLINK_IP=<gateway-IP>`
to `make fpga-run` if required by the board connection; the generated hardware
uses EDCL address `192.168.1.15`. To run the CPU baseline on the same image:

```sh
make -C "$soc" fpga-run TEST_PROGRAM="$deploy/$kernel-cpu.exe"
```

Repeat with `syr2k`, `trmm`, `symm`, `syrk`, `gesummv`, and `doitgen`, programming
the matching image each time. Record `TEST PASSED`, max error/tolerance and
the profiling table. Each bitstream must be paired with its own executable.

## Remaining unsupported cases

BICG and MVT generate LLVM and host objects but fail in Bambu's virtual-memory
SSA analysis. The debugger found a dependency crossing generated process
boundaries; see [ESPBambuMultiProcessFailure.md](ESPBambuMultiProcessFailure.md)
and the campaign's `bicg-mvt-reproducer.tar.gz`. No fix to Bambu is claimed by
this campaign. THREEMM, GEMVER, ATAX and TWOMM remain blocked earlier during
multibuffering; they are outside these seven FPGA builds.
