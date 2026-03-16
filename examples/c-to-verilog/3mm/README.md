# C to Verilog Example

Bambu HLS can convert C code to Verilog. This example demonstrates that flow with the 3mm benchmark. The Makefile is configured to generate Verilog and run a simulation to verify the results.

**This flow bypasses the MLIR passes and LLVM IR optimizations**. As a result, the generated Verilog is influenced only by the C implementation and the pragmas used in the code.


## What is needed?

The kernel code is in `forward_kernel.c`. The testbench code is in `forward_kernel_testbench.c`. The Makefile is configured to use these files to generate the Verilog and run the simulation.


## How to run?

```
make
```

Which by default will generate the verilog file.
To run the simulation, you can edit the `Makefile` to specify the target as the results file, which will trigger the simulation step:

```bash
# TARGET=$(ODIR)/bambu/baseline/06_verilog.v
TARGET=$(ODIR)/bambu/baseline/07_results.txt
```

Then run `make` again to execute the simulation which will verify the final verilog memory state against the memory state of a CPU execution of the C code. This will aslo provide a report on the number of cycles taken to execute the generated kernel.
