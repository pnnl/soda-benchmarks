# ResNet-18 reusable operator validation

Pretrained batch-one float32 ResNet-18 with 224×224 input compiles to 30
sequential calls on one engine per type: 20 convolutions, 9 additions and
1 GEMM. Twenty BatchNorms are folded and 17 ReLUs fused; the CPU performs
input layout conversion and two pooling operations. All three engines share
one ESP socket. Runtime dimensions/addresses are independent of hardware tiling.

The model uses torchvision `ResNet18_Weights.IMAGENET1K_V1` and deterministic
normal input (seed 1). Validation checks equivalence to unfused PyTorch on
1000 logits; this is not a measurement of ImageNet classification accuracy.
The maximum absolute error is `5.781650543212891e-6`, with top-one index 107.

## Correctness and FPGA performance

| Check | Result |
| --- | --- |
| Integrated plugin and graph tests | 67 lit + 7 graph tests pass |
| Individual operator full-system RTL | Conv/add/GEMM: 3/2/2 launches, 60/143/6 outputs, zero error |
| Operator RTL cosimulation with stalls | All three pass; wider-channel convolution also passes |
| Complete network native host/operator oracle | All 1000 outputs pass |
| Complete network on VC707, sequential engines | All 30 offloads and 1000 outputs pass |
| Complete network on VC707, LCS engines | All 30 offloads and 1000 outputs pass |

The full network needs a 102,308,928-byte shared arena. Individual operators
are tested in the 16 MiB full-system simulation fixture; the full model uses
the FPGA board's 1 GiB memory map. These are distinct validation scopes.

| Complete inference at 50 MHz | CPU cycles | Seconds |
| --- | ---: | ---: |
| Historical scalar ESP CPU-only runtime | 90,055,418,413 | 1801.1084 |
| Historical sequential accelerator engines | 7,669,179,138 | 153.3836 |
| LCS engines, identical preserved executable | 5,803,106,935 | 116.0621 |
| LCS engines, freshly compiled software | 5,804,799,476 | 116.0960 |

With identical software, LCS improves complete inference by **1.322×**, reducing
time by **24.33%**. The fresh integrated run is 15.51× faster than the historical
scalar CPU baseline, which was not rerun in this integration campaign. That
CPU baseline uses scalar C++ loops, GCC `-O2 -ffp-contract=off`, and cacheable
storage; it does not use a tuned BLAS or optimized CPU inference library.
Timed inference includes host/runtime work, copies, pooling and offloads;
program loading, boot and final numerical validation are outside the region.

The LCS complete SoC closes timing at +0.098357 ns setup / +0.053293 ns hold.
Its bitstream SHA256 is
`6762fda9b5391d51bff04b536df89af0236109fb3ce91df39996095e37591f03`.
This is a measured schedule comparison with fixed local resources, not a
completed search for optimal tiling, banking or engine count.

Use the [emilia tutorial](ESPSynthesisSimulation.md) for setup and individual
operator synthesis/simulation, and the
[neural-operator guide](../examples/esp-validation/reusable-nn/README.md) for
full-model compilation, FPGA software and board execution. The
[integration report](ESPReuseLCSIntegration.md) records controlled artifact
identities, resource comparisons and immutable raw evidence links. The old
sequential baseline and new LCS build remain preserved in separate checkouts.
