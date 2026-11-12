# EELS through reusable ESP operators

This local extension follows the same TOSA → generated host/runtime + reusable
hardware → LLVM → Bambu → ESP path as ResNet-18. The source model uses native
PyTorch Conv1D. `scripts/reusable_nn/prepare_model.py` extracts its channels,
kernel, stride, padding and dilation and converts dense Conv1D into equivalent
height-one Conv2D for the installed Torch-MLIR exporter. The original model is
kept intact and both representations are compared before export.

The 240-bin local reference encoder becomes four calls on convolution instance
0 and two calls on GEMM instance 0. All four LeakyReLUs are fused into hardware;
there is no CPU activation fallback. Two layout transposes execute on the CPU.
Both `z_mean` and `z_log_var` remain graph outputs. The generated C ABI writes
their flattened values consecutively to its output pointer; `plan.json.outputs`
records each shape and offset. This does not merge or discard either dense head.

Weights are loaded from the reference fixture, externalized to `parameters.bin`
and placed in shared DRAM. They are not hardware constants. The fixture's training
provenance is unknown. Its 240-bin input must not be silently substituted for the
480-bin input mentioned in the application slides.

## Local validation, 10 November 2026

| Check | Result |
| --- | --- |
| PyTorch encoder against supplied golden outputs | PASS; max error 1.19e-7 |
| Generated host and native operator runtime | PASS; both heads, max error 7.45e-8 |
| Synthesized RTL with DMA stalls | PASS; all six hardware outputs match the runtime oracle |
| Synthesized RTL with 97-cycle transaction latency | PASS; both final heads |
| Full EELS in full-system ESP RTL | PASS; six launches, 32 outputs, max error approximately 7.5e-8 |
| Different-shape/slope fixture in full-system RTL | PASS; six launches, ten outputs, max error approximately 6e-8 |
| Existing residual fixture on the same RTL | PASS; ordinary ReLU, addition, pooling and one output |
| Compiler / frontend and graph regressions | 68 / 13 tests pass |

The complete encoder uses 382848 shared bytes including its descriptor. The
full-system run records 5069323 CPU cycles including runtime/setup/copies and
offloads; this is not an FPGA wall-time or speedup measurement. No FPGA was
programmed. See `validation.json` for artifact hashes and exact evidence paths.
Failed pre-PMA-correction runs remain preserved and are not counted as passing.

## Per-offload programming

The transport is the existing DMA descriptor, plus ESP's normal start/done MMIO
registers. This change does not add a bank of custom MMIO configuration registers.
The engine latches a fresh descriptor on every launch. `plan.json` contains named
`configuration` fields and the selected descriptor version for each convolution.

| Invocation parameters | Hardware choices |
| --- | --- |
| N, input/output spatial dimensions, input/output channels | Tile pixels, output lanes, local channel tile |
| Kernel dimensions, stride, padding, dilation | Local storage and banking |
| Input, weight, bias and output locations | Sequential or LCS schedule |
| Activation mode and LeakyReLU slope | Floating-point arithmetic implementation |

The authoritative wire format is `ReusableOperatorABI.h`. Existing convolution
fields retain their positions. Slot 17 is activation mode (0=none, 1=ReLU,
2=LeakyReLU); slot 22 stores float32 slope bits. LeakyReLU uses descriptor version
2. The new hardware accepts v1 descriptors for none/ReLU and v2 descriptors for
the extended activation contract; v1 hardware cannot execute v2 offloads.
The public runtime uses byte offsets into arena data; descriptor addresses are
32-bit word offsets including the 256-byte header. Each field and float occupies
the low word of an eight-byte DMA slot. GEMM and addition retain their v1 ABI.
Supported LeakyReLU slopes are finite float32 values in [0,1].

`lower.py` recognizes the export pattern `max(x,0) + min(x,0)*alpha`, moves it
through layout views, and fuses it only into a single-use convolution. A graph
that also exposes the unactivated result is rejected rather than changed.
Constant zero spatial padding is folded into the convolution descriptor.
Grouped Conv1D and unsupported padding modes are rejected explicitly.

## Reproduce on emilia

The isolated checkout/container is `soda-eels-reuse-local-20261110`, based on the
four-commit review branch. Nothing here has been pushed or installed in Tommaso's
checkout. The older local FFN experiments are separate and preserved.

Container mounts: this checkout at `/workspace`, the existing Python dependency
environment at `/venv`, and the read-only reference
`/home/william/pnnl/eels/eels_sysc_catapult` at `/eels-reference`.
Set `MLIR_PLUGIN_PATH=/workspace/build/plugin/lib/SODAPlugin.so`,
`CLANG=/usr/bin/clang-19`, `RISCV_CLANG=/usr/bin/clang-19`, and include the
MLIR/Torch-MLIR Python packages plus `/venv/lib/python3.11/site-packages` in
`PYTHONPATH`, as in the existing container.

Inside that container, from `/workspace`:

```bash
cmake -S examples/soda-plugins -B build/plugin -G Ninja \
  -DMLIR_DIR=/opt/llvm-project/lib/cmake/mlir \
  -DLLVM_DIR=/opt/llvm-project/lib/cmake/llvm \
  -DSODAP_BAMBU_ROOT=/opt/panda -DCMAKE_BUILD_TYPE=Release
cmake --build build/plugin -j6
python3 models/pytorch/eels/export_reusable.py /eels-reference build/my-eels
bash scripts/tosa_to_reusable_nn.sh build/my-eels/01_tosa.mlir build/my-eels/compiled
python3 models/pytorch/resnet18/run_native.py build/my-eels --atol 1e-6 --rtol 1e-5

NN_CONV_SCHEDULE=lcs NN_GEMM_SCHEDULE=lcs \
  bash scripts/reusable_nn/generate_hardware.sh build/my-eels-hardware
python3 scripts/reusable_nn/combine_rtl.py build/my-eels-hardware
bash examples/esp-validation/reusable-nn/build-cosim.sh build/my-eels-hardware build/my-eels-cosim
SODAP_RTL_STALL_MASK=7 python3 models/pytorch/resnet18/run_native.py build/my-eels \
  --library host-only.so --runtime build/my-eels-cosim/rtl-runtime.so \
  --backend rtl-stalls --atol 1e-6 --rtol 1e-5
python3 examples/esp-validation/reusable-nn/stage_app.py build/my-eels build/my-eels-app
```

The `run_native.py` path is historical; this is the same generic generated-host
runner used for ResNet. In RTL mode, each hardware output is checked independently
against the runtime oracle, then consumed by subsequent graph operations. The
final two heads are compared with the supplied golden outputs. DMA backpressure,
output bounds, number of writes and DMA completion are checked too.

On the host, from this checkout, for a fresh full-system simulation:

```bash
export ESP_HANDOFF_ROOT=/home/william/bambu-esp-example/validation/esp-local-checkpoint-20261106/machine-handoff
source examples/esp-validation/emilia/env.sh
bash examples/esp-validation/reusable-nn/build-software.sh build/my-eels-app "$ESP_ROOT" "$ESP_SOC"
NN_TIMEOUT_CYCLES=50000000 ESP_SIM_TIME_MS=2000 \
  bash examples/esp-validation/reusable-nn/run-system.sh \
  build/my-eels-hardware build/my-eels-app "$ESP_SOC" build/my-eels-system
```

This runs the generated RISC-V application, ESP CPU, NoC, DMA and synthesized
engines. It does not program an FPGA. EELS fits the 16 MiB simulation memory map.

The system runner compiles a private copy of `tile_cpu.vhd` with Ariane's
cacheable region limited to the lower half of the configured simulation RAM.
The preserved ESP tile otherwise assumes a 512 MiB cacheable region, even when
the fixture only has 16 MiB RAM. That mismatch makes CPU reads return stale
values when hardware overwrites a previously CPU-read tensor. The corrected
fixture puts the DMA arena at 0x80800000 outside the 8 MiB cacheable region;
`pma.json` and source hashes record the configuration. The shared ESP tree and
FPGA configuration are unchanged. Optional `NN_TRACE=1` during software build
prints raw descriptors and initial output words for diagnosing such mismatches.

Additional regressions use `export_fixture.py`: a different model with dilation,
channel tails, nonzero biases, slopes 0.125/0.625 and differently sized outputs.
It runs on exactly the same hardware. The existing residual fixture checks old
ReLU descriptors, addition, pooling and single-output compatibility.

```bash
python3 -c 'from lit.main import main; main()' -v build/plugin/test
python3 -m unittest discover -s scripts/reusable_nn -v
```

This implements bounded static-shape float32 inference and runtime-configured
hardware reuse. It does not add architecture search, dynamic Python shapes,
concurrent offloads, P2P streaming or persistent weights across network calls.
