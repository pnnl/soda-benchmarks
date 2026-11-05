# SMALL PolyBench FPGA campaign

This campaign uses the reconciled PR17 flow and the MLIR fixes consolidated in
`0081c52`, on `local/esp-lcs-small-ready`. It uses the unchanged container Bambu
binary; the candidate Bambu patch is kept on the investigation branch.

All eleven candidates generate SMALL LLVM, RISC-V objects and Bambu RTL with
the default `baseline` flow. Ten pass the unchanged native numerical check:
GEMM, SYR2K, TRMM, SYMM, SYRK, GESUMMV, DOITGEN, THREEMM, ATAX and TWOMM.
GEMVER fails 76/120 outputs with maximum absolute error 0.09375 versus 0.001
tolerance and is excluded from the deployable-image campaign. BICG and MVT
remain blocked by the separate Bambu compiler issue.

## Generation and validation

For each candidate, the configured SODA container runs:

```sh
sb-cli init --base_dir "$CAMPAIGN/generated" --output_dir "$kernel" \
  --benchmark_name "$kernel" --dataset SMALL --dtype float32 \
  --backend esp --stage object --device xc7vx485t-2ffg1761 --clock_period 20
make -C "$CAMPAIGN/generated/experiments/$kernel" \
  LCS_PROFILE=true LCS_DMA_WORD_STRIDE=2 all cpu-baseline-object
```

The 64-bit-slot layout and profiler are enabled in both host and accelerator
generation. Native checks use the generated CPU LLVM and original test data;
neither the reference values nor the numerical tolerance are changed.

After generation, with the installed ModelSim/Vivado and RISC-V environments:

```sh
export LCS_TIMEOUT_CYCLES=20000000 ESP_SIM_TIME_MS=2000
export ESP_INSTRUCTION_TRACE=0
python3 examples/esp-validation/generated-lcs/run-campaign.py \
  "$CAMPAIGN" "$ESP_ROOT" "$COMPILED_SOC" --flow baseline \
  --kernels doitgen threemm gemm syr2k trmm symm syrk gesummv atax twomm \
  --sim-jobs 6 --fpga-jobs 2 --adapter-timeout 7200 --system-timeout 28800 \
  --cpu-build-only
```

Each accelerator must pass the numerical adapter tests, including repeated
invocations and backpressure, and a full-system RISC-V/ESP simulation before
Vivado runs. Implementation checks setup/hold timing and the expected clocks.
The watchdog limits are larger for SMALL workloads but remain finite. The
wrapper watchdog change lies entirely within `synthesis translate_off`;
the synthesizable wrapper text is unchanged.

Passive CPU instruction traces are redirected to `/dev/null` to save disk
space. UART output, DMA protocol checks, numerical results and profiler
output remain enabled. CPU baseline executables are linked for board use,
after native numerical validation; `--cpu-build-only` does **not** claim a
full-system CPU-baseline simulation pass. Their status is `BUILT`, not `PASS`.
The collector makes this distinction explicit:

```sh
python3 examples/esp-validation/generated-lcs/collect-results.py "$CAMPAIGN" \
  --allow-built-cpu-baseline --require-complete
```

Without `--allow-built-cpu-baseline`, its original requirement for a passing
CPU full-system simulation remains in place. With the flag, readiness requires
a passing native numerical check, a linked RISC-V ELF with no undefined symbols,
and the passing accelerator full-system simulation and FPGA checks.

## Local artifacts and status

The campaign completed on November 5, 2026: all ten selected kernels passed
native numerical checks, adapter tests and full-system RTL simulation, and
produced VC707 bitstreams with positive setup and hold slack. The final
collector reports **10/10 ready for board**. No new board execution is claimed.

The campaign is under `local-results/vc707-small-20261104/` in
`/home/william/bambu-esp-example/experiments/soda-pr17-reconciled`.
`status.json` records simulation and FPGA stages; each kernel also
has `generation.json`, generation/native logs, generated-input checksums,
isolated ESP files and implementation reports. Deployment files are in
`<kernel>/deploy/`: `<kernel>.bit`, `<kernel>.exe`, `<kernel>-cpu.exe`,
`prom.bin`, `prom.srec` and `SHA256SUMS`. Each application must be used with
its matching fixed-shape bitstream.

The original validated MINI campaign is preserved. No board is programmed by
this campaign. The generated LLVM, objects, RTL and initialization files retain
their generation-time checksums. The collector also verifies the installed RTL,
the Vivado project inputs and the correspondence between the simulated and
implemented SoC files.

## Results

These are full-system RTL simulation cycle counts, with the CPU and accelerator
at 50 MHz and the NoC/memory at 100 MHz. L2/LLC caches are disabled and DMA is
noncoherent. The numerical threshold is **1,000,000 in the error column's
units**, equivalent to absolute error 0.001.

`Kernel cycles` includes allocation, packing, accelerator execution and
unpacking. `Hardware cycles` measures the accelerator boundary, including DMA
and memory stalls; it is part of the kernel interval, not an additional cost.
WNS and WHS are the implemented SoC's worst setup and hold slack.

| Kernel | Max error (10^-9) | Kernel cycles | Hardware cycles | WNS (ns) | WHS (ns) |
|---|---:|---:|---:|---:|---:|
| gemm | 11,444 | 5,513,088 | 578,239 | 0.629 | 0.050 |
| syr2k | 7,629 | 7,319,814 | 1,150,982 | 0.515 | 0.052 |
| trmm | 7,629 | 4,861,474 | 794,386 | 0.586 | 0.052 |
| symm | 30,518 | 6,062,308 | 1,111,886 | 0.541 | 0.054 |
| syrk | 5,722 | 5,886,199 | 597,966 | 0.578 | 0.050 |
| gesummv | 15,259 | 3,574,948 | 100,748 | 0.506 | 0.051 |
| doitgen | 954 | 12,170,999 | 966,643 | 0.885 | 0.050 |
| threemm | 3,815 | 7,059,949 | 2,708,472 | 0.400 | 0.051 |
| atax | 30,518 | 7,617,983 | 4,459,079 | 0.613 | 0.050 |
| twomm | 183,105 | 6,478,169 | 2,193,987 | 0.469 | 0.052 |

CPU baseline binaries have native numerical validation and successful RISC-V
link checks. Their board timings remain to be measured, so this table does not
claim CPU-to-accelerator speedups.

The [evidence directory](validation/esp-small-20261104/) contains per-kernel
adapter logs and full-system UART extracts, with hashes of the complete
simulator transcripts. ESP ends a successful simulation with
`Failure: Program Completed!`; this is the testbench's completion marker.
The runner additionally requires `TEST PASSED` and rejects fatal errors,
unexpected failures and timeouts.

- [Results CSV](validation/esp-small-20261104/results.csv): profiling regions and timing margins.
- [Results JSON](validation/esp-small-20261104/results.json): validation status, resource use, clocks and deployment checksums.
- [Provenance](validation/esp-small-20261104/provenance.json): commits, compiler fingerprint and generated-input hashes.
- [Excluded kernels](validation/esp-small-20261104/excluded.json) and [GEMVER native log](validation/esp-small-20261104/gemver-native.log): the two Bambu failures and unchanged numerical failure kept outside this campaign.
