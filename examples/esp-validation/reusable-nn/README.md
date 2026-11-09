# Reusable operators for ResNet 18

This flow compiles pretrained ResNet-18 inference into calls to one reusable
convolution engine, one elementwise addition engine, and one GEMM engine. The
three engines share one ESP socket and execute sequentially. Model dimensions,
weights and tensor locations are runtime arguments; hardware tiles and lane
counts belong to a separate hardware-generation step.

The standard test uses torchvision `ResNet18_Weights.IMAGENET1K_V1`, float32,
batch one and a 224×224 input. The input is a deterministic normal tensor for
numerical validation, so the results do not measure ImageNet classification
accuracy. The reference comes from the original pretrained PyTorch model.

## Compilation and execution

`models/pytorch/resnet18/export_reusable.py` folds each inference BatchNorm into
its convolution and exports actual TOSA MLIR through torch-mlir. The reusable
lowerer parses and verifies that MLIR, propagates NHWC layouts, folds constant
transposes, fuses single-use ReLUs, and plans storage from tensor lifetimes.
Residual tensors stay live until their addition consumes them.

The emitted host MLIR contains explicit operator, instance, invocation and
execution attributes. It lowers through the LLVM dialect to LLVM IR and then
native or RV64 machine code. `compiled/plan.json` records the mapping and
byte offsets. Constants are externalized to `parameters.bin`, independent of
the host executable and hardware schedule.

The full graph contains 20 convolution calls, 9 addition calls (8 residuals and
the classifier bias), and 1 GEMM. Twenty BatchNorms are folded and 17 ReLUs are
fused. The CPU performs the input NCHW-to-NHWC conversion, max pooling and global
average pooling. No intermediate activation is copied out of the shared arena
between accelerator calls.

The 256-byte descriptor and every float occupy 64-bit DMA slots with a zero
upper word. Public runtime addresses are byte offsets into the data area;
the runtime adds the descriptor prefix. The network data area uses
102,308,672 bytes and the complete arena uses 102,308,928 bytes. This padded
transport is inherited from the validated ESP adapter and can be optimized
separately.

## Operator contracts and hardware schedules

| Engine | Per-invocation values | Initial fixed hardware schedule |
| --- | --- | --- |
| Convolution | N, input/output spatial sizes, input/output channels, kernel size, stride, padding, dilation, A/W/bias/C locations, ReLU flag | 4 spatial positions, 8 output-channel lanes, 16 input channels per tile |
| Addition | Element count, A/B/C locations, ReLU flag | Blocks of 128 elements |
| GEMM | M/N/K, A/B/C locations | 4×4×4 tiles |

Convolution consumes NHWC activations and OHWI weights. GEMM uses contiguous
row-major operands with alpha=1 and beta=0. Channel, spatial and matrix tails
are handled at runtime. Float32 arithmetic and each engine's local resources
remain fixed for a hardware build. The initial schedules establish functional
reuse; they are not a performance-tuned ResNet architecture.
The convolution generator gives each output lane separate weight and accumulator
banks. The default schedule executes loads, computation and stores sequentially.
Set `NN_CONV_SCHEDULE=lcs` and `NN_GEMM_SCHEDULE=lcs` for concurrent load,
compute and store tasks using the shared Dataflow ping-pong transformation.
Convolution retains its per-output-lane weight banks and accumulation order.
The host still issues one operator invocation at a time; overlap occurs inside
an engine, without changing the operator descriptor or host invocation plan.

`ReusableOperatorABI.h` defines the wire fields shared by the runtime,
MLIR hardware generators and RTL dispatcher's generated constants. Each
descriptor selects one engine. The host waits for DMA completion before
issuing the next call. The dispatcher does not reset the engines between calls.

The initial compiler accepts static positive f32 tensors of rank 1–4, one input
and one output, batch-one GEMM, equal-shape addition and single-use conv/add
ReLU fusion. It rejects unsupported TOSA operations and broadcast addition.
The reusable hardware accepts new valid dimensions without resynthesis;
compiling a different static graph produces a different host invocation plan.

## Reproduce the model and hardware

Use the SODA container with PyTorch, torchvision, torch-mlir, LLVM and Bambu.
The extended plugin must be built from this checkout. For example:

```bash
source /opt/panda/settings.sh
export PATH=/workspace/scripts:/opt/soda-opt/bin:/opt/llvm-project/bin:$PATH
export PYTHONPATH=/opt/llvm-project/python_packages/mlir_core:/opt/torch-mlir/python_packages/torch_mlir:/opt/soda-opt/python_packages/soda:/workspace
export TORCH_HOME=/workspace/build/torch-cache
export MLIR_PLUGIN_PATH=/workspace/build/resnet18/plugin/lib/SODAPlugin.so

cmake -S examples/soda-plugins -B build/resnet18/plugin -G Ninja \
  -DMLIR_DIR=/opt/llvm-project/lib/cmake/mlir \
  -DLLVM_DIR=/opt/llvm-project/lib/cmake/llvm \
  -DSODAP_BAMBU_ROOT=/opt/panda -DCMAKE_BUILD_TYPE=Release
cmake --build build/resnet18/plugin --target SODAPlugin -j4

python3 models/pytorch/resnet18/export_reusable.py build/resnet18
bash scripts/tosa_to_reusable_nn.sh build/resnet18/01_tosa.mlir build/resnet18/compiled
python3 models/pytorch/resnet18/run_native.py build/resnet18
bash scripts/reusable_nn/generate_hardware.sh build/resnet18/hardware
```

Hardware generation takes no model input. Its optional variables are
`NN_CONV_PIXELS`, `NN_CONV_LANES`, `NN_CONV_CHANNELS`, `NN_ADD_BLOCK`, and
`NN_GEMM_M/N/K`, `NN_CONV_SCHEDULE`, and `NN_GEMM_SCHEDULE`. Schedule values
are `sequential` (default) or `lcs`. `NN_SYNTHESIZE=0` stops after generating
LLVM IR. Changing these variables leaves the compiled host and parameter blob
unchanged. Addition currently uses its sequential implementation.

On the host with Verilator and a C++ compiler:

```bash
python3 scripts/reusable_nn/combine_rtl.py build/resnet18/hardware
bash examples/esp-validation/reusable-nn/build-cosim.sh build/resnet18/hardware build/resnet18/cosim
```

With NumPy available, execute the generated host against real RTL:

```bash
python3 models/pytorch/resnet18/run_native.py build/resnet18 \
  --library host-only.so --runtime build/resnet18/cosim/rtl-runtime.so \
  --backend rtl-operator-cosim
```

This co-simulation uses a native CPU host, the generated RTL and the actual
ESP DMA64 adapter. It checks every accelerator output against the C++ operator
oracle, poisons the destination before RTL execution, and passes the RTL result
to the next graph node. Final logits are checked against unfused PyTorch.
It is distinct from the full-system tests below. The full model simulates
billions of accelerator cycles and can take many hours.

## Full-system RTL tests for each operator

Each test uses compiler-generated RV64 host code, the real bare-metal runtime,
ESP CPU, NoC, socket, DMA and memory model. The three tests share the same
three-engine RTL build, but each application invokes only its selected type:

- Convolution: 7×7 stride two, dilated 3×3, then 1×1; channel counts 3→9→11→5,
  padding, channel tails and fused/unfused ReLU.
- Addition: 143 elements, crossing the 128-element block boundary, repeated
  with a reshaped tensor and a changed ReLU flag.
- GEMM: 3×5 times 5×7, then 3×7 times 7×2, including all tile tails.

Passing `--wide-channels` to `export_operators.py` changes convolution channels
to 3→19→23→5, exercising multiple input-channel tiles as well as tails for the
default 16-channel hardware. Use a separate output directory for this fixture.
For native-host/RTL-accelerator cosimulation, `SODAP_RTL_STALL_MASK=1` or `7`
inserts randomized DMA stalls. This differs from the initial `NN_STALL` delay
used by full-system tests.

Export and compile inside the container:

```bash
python3 examples/esp-validation/reusable-nn/export_operators.py build/resnet18/operators
for kind in conv add gemm; do
  model=build/resnet18/operators/$kind
  bash scripts/tosa_to_reusable_nn.sh "$model/01_tosa.mlir" "$model/compiled"
  python3 models/pytorch/resnet18/run_native.py "$model"
  python3 examples/esp-validation/reusable-nn/stage_app.py "$model" "$model/esp-app"
done
```

On the host, set `ESP_ROOT`, `ESP_SOC`, `ESP_MODELSIM_INI` and optionally
`ESP_MODELSIM_WORK` to the validated ESP source, compiled SoC and vendor library
configuration. Put the licensed ModelSim tools and RISC-V toolchain on `PATH`.
Set `LD_LLD=ld.lld-16` when the older GNU linker cannot read Clang ISA attributes.
Then, for each `kind`:

```bash
model=build/resnet18/operators/$kind
bash examples/esp-validation/reusable-nn/build-software.sh "$model/esp-app" "$ESP_ROOT" "$ESP_SOC"
bash examples/esp-validation/reusable-nn/run-system.sh \
  build/resnet18/hardware "$model/esp-app" "$ESP_SOC" "$model/system"
```

Use a fresh simulation output directory. `ESP_RECOMPILE_SOC_TOP=1` refreshes
the top against reconstructed vendor libraries. Each `full-system/validation.json`
requires numerical success, the expected CPU operator sequence, matching
physical RTL launch/completion counts and normal ESP program completion.
`NN_STALL` inserts initial DMA stalls through the existing adapter.

## Complete network software for the FPGA

Stage the full model with `stage_app.py`, then build with
`NN_DRAM_BYTES=1073741824`. This selects the board's 1 GiB physical memory map
and starts accelerator allocations at `0xa0200000`, after the Ethernet-reserved
area. The build checks that the program, weights and stack remain separate
from the DMA arena. The generated ELF and raw image are `nn_app.exe` and
`nn_app.bin`; the software contains the same input, pretrained parameters and
1000 reference logits used in native validation.

```bash
python3 examples/esp-validation/reusable-nn/stage_app.py build/resnet18 build/resnet18/esp-app
NN_DRAM_BYTES=1073741824 bash examples/esp-validation/reusable-nn/build-software.sh \
  build/resnet18/esp-app "$ESP_ROOT" "$ESP_SOC"
python3 examples/esp-validation/reusable-nn/stage_hardware.py \
  build/resnet18/hardware build/resnet18/esp-package
python3 examples/esp-validation/generated-lcs/prepare-soc.py \
  build/resnet18/esp-package "$ESP_ROOT" "$ESP_SOC" build/resnet18/esp
```

The packaged hardware mechanically combines the same namespaced Bambu RTL and
dispatcher used in simulation. Build the isolated SoC after operator validation:

```bash
bash examples/esp-validation/reusable-nn/build-fpga.sh \
  build/resnet18/operators system build/resnet18/esp/socs/vc707 build/resnet18/fpga
```

This wrapper checks all three full-system results and their input hashes before
calling `generated-lcs/build-fpga.sh`. If routing leaves a timing violation,
`close-timing.tcl` attempts physical optimization of the routed checkpoint
without changing the RTL or clocks:

```bash
vivado -mode batch -source examples/esp-validation/reusable-nn/close-timing.tcl \
  -tclargs build/resnet18/esp/socs/vc707/vivado/esp-xilinx-vc707-xc7vx485t.runs/impl_1/top_routed.dcp \
  build/resnet18/fpga-physopt
```

It writes a bitstream only when setup and hold slack are nonnegative. With the intended board reserved and a
UART recording already running, `run-board.py` programs the specified JTAG
target, loads the full ELF, resets the board and checks all 30 runtime calls
and all 1000 output comparisons through UART. For the configured VC707 setup:

```bash
python3 examples/esp-validation/reusable-nn/run-board.py \
  build/resnet18/esp/socs/vc707 build/resnet18/esp-app \
  build/resnet18/fpga-physopt/top.bit build/resnet18/board \
  --uart-host william@espdev --uart-log /home/william/resnet18-uart.log \
  --jtag-host espdev --jtag-port 3131 \
  --esplink-ip 128.59.22.75 --esplink-port 46313
```

Synthesis reports alone do not constitute a placed-and-routed bitstream or an
FPGA execution result.

For the configured `vc707-03` serial server, port 4330 on `espdev` records UART
at 38400 baud. `capture_uart.py localhost 4330 NEW_LOG` can run on that server
with Python 3.12 to create a fresh bounded recording. Pass its path to
`run-board.py --uart-log`; preserve existing recordings. The board runner also
checks positive setup and hold slack before programming.

The existing ESP bare-metal allocator does not reclaim allocations; the board
application performs one inference per reset. Repeated inferences within one
boot will need persistent model/context lifetime support or a reclaiming arena.

## CPU-only comparison on ESP

Build a separate application with `NN_EXECUTION=CPU_ONLY` to execute all operators
on the RISC-V CPU. It reuses the generated host object, folded weights, input,
reference and invocation plan. Convolution, addition and GEMM use the runtime's
scalar C++ loops, compiled with GCC `-O2 -ffp-contract=off`; accelerator probing,
DMA setup and MMIO launches are disabled. Pooling and transpose also run on the
CPU. The runtime retains its eight-byte tensor slots and allocates the arena in
cacheable BSS in the lower half of DRAM. The linker checks that this arena, the
weights and stack fit. This is a software baseline for the current runtime,
without BLAS or CPU-specific tiling/vectorization.

```bash
python3 examples/esp-validation/reusable-nn/stage_app.py build/resnet18 build/resnet18/esp-cpu-app
NN_EXECUTION=CPU_ONLY NN_DRAM_BYTES=1073741824 \
  bash examples/esp-validation/reusable-nn/build-software.sh \
  build/resnet18/esp-cpu-app "$ESP_ROOT" "$ESP_SOC"
python3 examples/esp-validation/reusable-nn/run-board.py \
  build/resnet18/esp/socs/vc707 build/resnet18/esp-cpu-app \
  build/resnet18/fpga-physopt/top.bit build/resnet18/board-cpu \
  --uart-host william@espdev --uart-log /home/william/resnet18-cpu-uart.log \
  --timeout 21600
```

Use a fresh staged directory and UART recording, with a capture duration longer
than the run timeout. `$ESP_SOC` for software compilation must contain the built
ESP driver archives. The runner reads `memory.json` to distinguish CPU-only and
accelerated execution, checks all 30 software operator completions, rejects any
accelerator launches during a CPU-only run, and validates all 1000 logits.

Both modes time `forward_kernel` with the machine cycle counter. This includes
runtime initialization, tensor copies and all graph operations, and excludes
program loading, boot initialization and the final output comparison. Divide
CPU-only cycles by accelerated cycles for the speedup on the same CPU clock.
The validated VC707 image drives its CPU from the 50 MHz reference clock.
After both runs pass, verify that they used identical host code, weights, input,
reference and bitstream, then generate the comparison:

```bash
python3 examples/esp-validation/reusable-nn/compare-board.py \
  build/resnet18/board-cpu/validation.json build/resnet18/board/validation.json \
  build/resnet18/cpu-vs-fpga.json --cpu-hz 50000000
```

If only the UART monitor stops while the board is still running, repeat the
original `run-board.py` command with `--resume`. This checks the saved artifact
hashes and resumes at the original UART offset without programming or resetting
the FPGA. A CPU-only run can also be identified by its explicit `NN_MODE` line
when the first startup line is lost, while still requiring the complete operator
sequence, output count and numerical check.
