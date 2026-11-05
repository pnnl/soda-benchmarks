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

The campaign is under `local-results/vc707-small-20261104/` in the reconciled
worktree. `status.json` records simulation and FPGA stages; each kernel also
has `generation.json`, generation/native logs, generated-input checksums,
isolated ESP files and implementation reports. Completed deployment files
will be in `<kernel>/deploy/`: the bitstream, offload executable, CPU baseline
executable, boot ROM files and checksums.

The original validated MINI campaign is preserved. No board is programmed by
this campaign. FPGA results are pending until the corresponding stages report
`PASS`; successful Bambu synthesis alone is not a claim of device fit or timing
closure.
