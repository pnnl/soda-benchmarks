# ESP PolyBench CLI validation — 17 September 2026

## Result for the PR review

The same `sb-cli init` + `make` workflow generated the CPU baseline and
runtime-offload application for **all 13 requested PolyBench kernels**, using
the benchmark name to select the kernel. No kernel-specific Python,
lowering-pass or host-code edits were needed. Shared campaign
support fixes were needed, as documented below; this is not an out-of-the-box
claim for the unmodified `abac36b` checkout.

There are **31 linked RISC-V executables**: 13 CPU, 13 runtime-offload and five
IR-marshalling variants. On VC707, **28 eligible variants completed: 25 passed
the original accuracy check and three failed**. Three other runtime variants
were built but intentionally not launched because of the original FFN DMA
geometry constraint. All 13 benchmark names were executed as CPU baselines;
this is **not** a claim that all 13 were tested with accelerator offload.

These experiments use the **existing FFN accelerator**, not hardware newly
synthesized from each Python benchmark by Bambu. Compiler generation ran in
Docker; ESP baremetal linking ran natively on Ubuntu. FPGA loading and minicom
recording were performed separately by Will.

This review update records the experiments and minor cleanup only. It does not
change the marshalling default, remove runtime conversion helpers, or implement
multi-offload IR support. Those functional review requests remain open.

## Coverage

All cases use **MINI, float32**, with unchanged absolute tolerance **`1e-3`**.
The runtime implements Q16.16 transfers to the existing FFN. Each kernel's
input/reference header is identical across its available variants. `bicg` and
`mvt` validate both returned tensors, not just the first output.

| Benchmark | Offloads in runtime variant | CPU on board | Runtime on board | IR + fusion on board |
| --- | ---: | --- | --- | --- |
| `syr2k` | 2 | PASS | PASS | Not generated |
| `gemm` | 1 | PASS | PASS | PASS |
| `threemm` | 3 | PASS | PASS | Not generated |
| `trmm` | 1 | PASS | PASS | PASS |
| `symm` | 1 | PASS | PASS | PASS |
| `syrk` | 1 | PASS | PASS | PASS |
| `gemver` | 2 | FAIL | FAIL | Not generated |
| `bicg` | 2 | PASS | Not run: DMA preflight | Not generated |
| `atax` | 2 | PASS | Not run: DMA preflight | Not generated |
| `gesummv` | 2 | PASS | Not run: DMA preflight | Not generated |
| `doitgen` | 1 | PASS | PASS | PASS |
| `twomm` | 2 | PASS | FAIL | Not generated |
| `mvt` | 2 | PASS | PASS | Not generated |

The eight missing IR variants have multiple offloads per function, which the
reviewed IR-marshalling implementation does not support. Runtime marshalling
does support the sequential offloads exercised here, including three for
`threemm`. The single-offload IR guard was not removed for these experiments.

`atax`/`bicg` have 1,596 input elements per relevant tile and `gesummv` has 900;
these are not multiples of the original FFN loader's eight-element consumption
group. Their runtime executables remain quarantined. Subsequent hardware DMA
changes do not retrospectively validate those three cases: rerunning their
preflight and testing them remains separate work.

The complete 31-artifact inventory, build return codes, final executable and
kernel-object hashes, numerical outcomes and profile measurements are in
[results.csv](validation/esp-polybench-20260917/results.csv). `NOT_RUN` is distinct
from FAIL. Empty timing fields mean unavailable/not instrumented, not zero cost.
All 31 final executables were checked for unresolved symbols; none remained.

## Reproducing the CLI generation

Use the compiler/plugin/container setup in
[ESPValidation.md](ESPValidation.md#compiler-and-plugin-setup). The frozen source
base was `abac36b8cefe5638733dd677916e283cafa9b154`, with companion `soda-opt`
commit `19377c05acd191f3ae4a639e3f827d3a0e533e00` and the campaign support below.
See [provenance.json](validation/esp-polybench-20260917/provenance.json) for
the image ID and captured input hashes.

### Shared support used by this campaign

The [campaign patch snapshot](validation/esp-polybench-20260917/campaign-support.patch) contains:

- Multi-output `testdata.h` generation, needed for `bicg` and `mvt`.
- RISC-V `medany` code generation, needed for globals at/above `0x80000000`;
  the original setting produced relocation failures in `symm` and `trmm`.
- Selection of the validation driver through `CPU_MAIN` and corrected rebuild
  dependencies for staged applications.

The archived [campaign driver](validation/esp-polybench-20260917/campaign_driver.c)
checks every output and detects NaN/Inf/unwritten results despite baremetal
`-ffast-math`. It keeps the original golden values and tolerance. The patch and
driver are reproduction artifacts; this documentation update does **not**
apply the patch to the PR's implementation files. The campaign did not require
per-benchmark accelerator lowering or runtime changes.

For the matching source base, inspect/apply the snapshot in a disposable
checkout, or use equivalent already-integrated fixes. Do not apply it blindly
on top of a newer implementation:

```bash
git apply --check docs/validation/esp-polybench-20260917/campaign-support.patch
git apply docs/validation/esp-polybench-20260917/campaign-support.patch
```

Then, in the configured container shell at the repository root:

```bash
set -euo pipefail
repo="$PWD"
export CPU_MAIN="$repo/docs/validation/esp-polybench-20260917/campaign_driver.c"
export RISCV_CMODEL=medany
validation_root=$(mktemp -d /tmp/soda-esp-polybench.XXXXXX)
kernels=(syr2k gemm threemm trmm symm syrk gemver bicg atax gesummv doitgen twomm mvt)

build_variants() {
  local mode="$1" flow="$2" recipe="$3"
  shift 3
  local kernel name experiment app
  mkdir -p "$validation_root/$mode"
  for kernel in "$@"; do
    name="ffn_${kernel}_${mode}"
    python3 -m sb_cli init \
      --base_dir "$validation_root/$mode" --output_dir "$name" \
      --benchmark_name "$kernel" --dataset MINI --dtype float32 \
      --flow "$flow" --backend esp --stage object --instrumentation "$recipe"
    experiment="$validation_root/$mode/experiments/$name"
    make -C "$experiment" SCRIPTS_DIR="$repo/scripts"
    app="$experiment/output/esp/$flow/esp-app"
    test -s "$app/kernel.o"
    test -s "$app/testdata.h"
    llvm-readelf -h "$app/kernel.o"
    llvm-nm --undefined-only "$app/kernel.o"
  done
}

build_variants cpu baseline esp "${kernels[@]}"
build_variants runtime transformed esp "${kernels[@]}"
build_variants ir transformed esp-ir gemm doitgen symm syrk trmm
```

This is the campaign's CLI pattern, collected into one loop for readability.
`init` creates the experiment; `make` performs generation and compilation.
`--backend esp` selects the RISC-V software target; `--flow baseline` still
executes the computation on the CPU. `--flow transformed` selects offload;
`esp` uses runtime marshalling and `esp-ir` emits conversion IR plus fusion.
Only the benchmark name/output name changes within a given variant group.

Compilation does not execute the board or establish numerical correctness.
Relocatable kernel objects retain expected runtime dependencies. Follow the
[ESP linking instructions](ESPValidation.md#esp-dependencies-outside-this-compiler-pr)
for each staged application, then require a nonempty final ELF and no undefined
symbols. The campaign's direct baremetal build used `rv64imafdc`, `lp64d` and
`-DTD_MODE=cpu`, `runtime` or `ir` as appropriate for the driver banner.
Keep the three quarantined runtime executables out of the launch list.

## Measured board results

The archived [full UART capture](validation/esp-polybench-20260917/polybench.log)
contains 11 earlier reports followed by the complete 28-case suite, starting
at line 159. The final sequence matches the timed launch records. There are no
missing final reports or nonfinite outputs, and the expected offload/profile
call counts match. A loader success was recorded only as `LOADED_UNVERIFIED`;
the PASS/FAIL results here come from the application reports, not loader exits.

The following numbers are taken from the final 28 reports in that log. Maximum
errors are absolute values; the acceptance threshold is `0.001`. All timings
are cycles, summed across the kernel invocation. Failed cases are retained.

| Benchmark | Mode | Result | Errors / elements | Maximum absolute error | Total cycles |
| --- | --- | --- | ---: | ---: | ---: |
| `gemm` | cpu | PASS | 0 / 500 | 0.000001907 | 581,228 |
| `gemm` | runtime | PASS | 0 / 500 | 0.000412941 | 1,879,284 |
| `gemm` | ir | PASS | 0 / 500 | 0.000412941 | 1,813,716 |
| `doitgen` | cpu | PASS | 0 / 960 | 0.000000238 | 312,959 |
| `doitgen` | runtime | PASS | 0 / 960 | 0.000079632 | 1,837,159 |
| `doitgen` | ir | PASS | 0 / 960 | 0.000079632 | 1,793,923 |
| `syrk` | cpu | PASS | 0 / 900 | 0.000001907 | 757,213 |
| `syrk` | runtime | PASS | 0 / 900 | 0.000367165 | 1,980,920 |
| `syrk` | ir | PASS | 0 / 900 | 0.000367165 | 1,847,450 |
| `symm` | cpu | PASS | 0 / 600 | 0.000015259 | 576,437 |
| `symm` | runtime | PASS | 0 / 600 | 0.000656128 | 1,931,091 |
| `symm` | ir | PASS | 0 / 600 | 0.000656128 | 1,790,180 |
| `trmm` | cpu | PASS | 0 / 600 | 0.000001907 | 498,731 |
| `trmm` | runtime | PASS | 0 / 600 | 0.000287056 | 1,856,180 |
| `trmm` | ir | PASS | 0 / 600 | 0.000287056 | 1,782,957 |
| `mvt` | cpu | PASS | 0 / 80 | 0.000001907 | 189,880 |
| `mvt` | runtime | PASS | 0 / 80 | 0.000341415 | 3,527,753 |
| `syr2k` | cpu | PASS | 0 / 900 | 0.000002861 | 1,418,546 |
| `syr2k` | runtime | PASS | 0 / 900 | 0.000549316 | 3,852,836 |
| `twomm` | cpu | PASS | 0 / 384 | 0.000007629 | 504,627 |
| `twomm` | runtime | FAIL | 274 / 384 | 0.001991272 | 3,439,507 |
| `threemm` | cpu | PASS | 0 / 352 | 0.000000119 | 738,414 |
| `threemm` | runtime | PASS | 0 / 352 | 0.000290930 | 5,093,658 |
| `gemver` | cpu | FAIL | 1 / 40 | 0.001220703 | 462,870 |
| `gemver` | runtime | FAIL | 40 / 40 | 0.354003906 | 3,805,711 |
| `atax` | cpu | PASS | 0 / 42 | 0.000003815 | 184,033 |
| `bicg` | cpu | PASS | 0 / 80 | 0.000001907 | 184,694 |
| `gesummv` | cpu | PASS | 0 / 30 | 0.000003815 | 86,498 |

### Offload profiling breakdown

Values are cycles accumulated across all offloads in that invocation. `—` means
no separate region was reported, not zero work. CPU-only runs have no offload
regions; their total cycles appear above. Unattributed cycles are the total
minus the reported pack/accelerator/unpack/epilogue regions.

| Benchmark | Mode | Pack | Accelerator + wait | Unpack | Epilogue | Unattributed |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `gemm` | runtime | 60,461 | 76,369 | 39,708 | 79,777 | 1,622,969 |
| `gemm` | ir | 57,222 | 76,339 | 34 | 61,469 | 1,618,652 |
| `doitgen` | runtime | 49,754 | 59,126 | 75,155 | 23,263 | 1,629,861 |
| `doitgen` | ir | 28,142 | 59,132 | 55,804 | 23,244 | 1,627,601 |
| `syrk` | runtime | 47,137 | 74,830 | 70,548 | 142,922 | 1,645,483 |
| `syrk` | ir | 43,565 | 74,845 | 34 | 106,842 | 1,622,164 |
| `symm` | runtime | 45,200 | 51,639 | 47,326 | 95,717 | 1,691,209 |
| `symm` | ir | 54,079 | 51,644 | 66 | 73,611 | 1,610,780 |
| `trmm` | runtime | 45,141 | 51,708 | 47,288 | 39,249 | 1,672,794 |
| `trmm` | ir | 57,581 | 51,704 | 66 | 63,755 | 1,609,851 |
| `mvt` | runtime | 145,050 | 98,223 | 7,800 | — | 3,276,680 |
| `syr2k` | runtime | 100,098 | 149,414 | 140,240 | — | 3,463,084 |
| `twomm` | runtime | 66,329 | 63,871 | 53,123 | — | 3,256,184 |
| `threemm` | runtime | 104,216 | 102,896 | 82,067 | — | 4,804,479 |
| `gemver` | runtime | 130,858 | 98,317 | 7,714 | — | 3,568,822 |

`atax`, `bicg` and `gesummv` runtime variants have no board measurements and are
not included in these tables. Their CPU measurements are included. The full
machine-readable inventory, including the three `NOT_RUN` entries, remains in
[results.csv](validation/esp-polybench-20260917/results.csv).

## Profiling interpretation and remaining work

The table includes complete application-kernel timings, not just accelerator
execution. Every offloaded MINI variant in this run was slower end-to-end than
its CPU baseline. For example, GEMM took 581,228 CPU cycles, 1,879,284 runtime
cycles and 1,813,716 IR/fusion cycles. Runtime GEMM has 1,622,969 cycles outside
the named pack/accelerator/unpack/epilogue regions; that overhead needs separate
investigation before claiming a speedup. An absent epilogue row is not zero CPU
work, and a tiny fused unpack region can mean work moved into a consumer loop.

These are single-run smoke-test/profile observations, not statistically robust
performance measurements. The next steps are multi-offload IR support, explicit
revalidation of the three DMA-quarantined cases, precision improvements for
the remaining accuracy failures, and attribution of the large runtime overhead.

The recorded local bitstream hash is
`f93f1ec08d81bc59f3f3224044d922c40831d1c0d4d613f675bd4848c7620863`.
This and the executable hashes identify local artifacts accompanying the manual
runs; they are not independent attestation of the programmed FPGA or UART
producer. The current results supersede the older numerical observations for
this tested hardware revision; they do not establish that all PR concerns,
all PolyBench offloads, or arbitrary Python-to-Bambu hardware generation are solved.
