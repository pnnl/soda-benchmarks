# Reusable GEMM validation

The workload is `H = A @ B; Y = H @ W`, with `(M,N,K)=(7,9,13)` then
`(7,5,9)`. Both calls use physical instance 0. The intermediate H remains in
the shared arena; the second call reads the first call's output without a
host copy or accelerator reset. All 98 H/Y elements pass with zero error.

The [emilia tutorial](ESPSynthesisSimulation.md) covers synthesis and full-system
execution. The [GEMM guide](../examples/esp-validation/reusable-gemm/README.md)
adds native capture and component tests. See the
[integration report](ESPReuseLCSIntegration.md) for immutable evidence links.

## Validated schedules

Both sequential and overlapping load/compute/store implementations pass:

- Two dependent GEMMs, with immediate DMA and 50,000-cycle initial stalls.
- Ten calls covering row strides, coefficients, accumulation and matrix tails.
- Full-system execution with the generated RV64 host, ESP CPU, NoC, DMA,
  accelerator and memory model: two launches and 98 correct outputs.
- The alternative LCS 2×8×3 tile implementation passes both component suites
  under both stall settings. Host LLVM and ABI remain identical.
- Bad magic and version descriptors followed by valid calls recover without
  reset or unintended output writes.

| Measurement, 4×4×4 tiles | Sequential | LCS |
| --- | ---: | ---: |
| Component accelerator cycles, two calls | 7,041 | 5,316 |
| Full-system accelerator boundary cycles, two calls | 12,991 | 10,425 |
| Complete kernel CPU cycles | 564,504 | 561,847 |

The complete small application improves by 0.47%; CPU setup/copies dominate.
Component cycle improvements are not whole-application speedups. The two
full-system measurements use identical software.

Runtime matrix dimensions, operand locations, row strides and coefficients
remain separate from hardware tile sizes, banking and schedule. The frontend
currently recognizes supported static f32 matmul graphs; it does not infer
contracts for arbitrary operators or accept arbitrary dynamic Python shapes.

The integrated compiler passes 67 lit tests and 7 neural-graph tests. Full
ResNet-18 FPGA results concern the three-engine neural-operator design, whose
GEMM contract is narrower than this standalone alpha/beta/row-stride interface;
see [ResNet validation](ESPResNet18Validation.md).
