## IP Integration

The instrumentation recipes in this folder are based on the following example from the PandA-bambu repository:

- https://github.com/ferrandi/PandA-bambu/tree/4249e579ce3e15265dfd35c427342016f51fb1a9/examples/IP_integration

Files inside this folder not marked with a specific license are under PandA-bambu license: https://github.com/ferrandi/PandA-bambu

Examples included: 

- assert
- change-location
- hw-counters
- vector-dot


### How to extend?

The IP integration requires multiple files to be present. It also requires triggering the right calls to the IPs from MLIR code.


### Files involved

Simple example describing how to integrate and verify existing IP with functions written in C that receives structs passed by pointers.

Hereafter a small description of files as described in the original example:

top.c: file to be compiled/synthesized by bambu.
module_lib.h: header that declares the interfaces to existing Verilog IPs.
module_lib.xml: XML file that describes interfaces of existing Verilog IPs.
module1.v: verilog of an existing synthesizable IP.
module1.c: C stub used to emulate the module1 IP in C.
module2.v: verilog of an existing synthesizable IP.
module2.c: C stub used to emulate the module2 IP in C.
printer1.v: verilog of an existing non-synthesizable IP.
printer1.c: C stub used to emulate the printer1 IP in C.
printer2.v: verilog of an existing non-synthesizable IP.
printer2.c: C stub used to emulate the printer2 IP in C.
sodaInstrAssertLessThen.c: C file that implements the sodaInstrAssertLessThen function.
sodaInstrAssertLessThen.v: Verilog file that implements the sodaInstrAssertLessThen function.
main_test.c: C testbench
constraints_STD.xml: resource constraint file passed to bambu to generate a Verilog design with just 1 my_ip module.
test.xml: XML file describing the testbench inputs. It is empty since we use the main_test.c as testbench generator.
bambu.sh: synthesis and simulation script. It requires Vivado RTL and Verilator to properly work.
