# Compile Torch model to LLVM

FILE_PATH=torchscript.py
TRANSFORM_PATH=transform.mlir
SCRIPTS_DIR=../../../../../scripts
# Make sure the output directory is not empty, many files will be generated.
# ODIR=.
ODIR=output

EXP_CWD := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
IPDIR=$(EXP_CWD)/IPs

# Edit to change the target
# TARGET=$(ODIR)/04_llvm.ll

# TARGET=$(ODIR)/bambu/baseline/06_verilog.v
# TARGET=$(ODIR)/bambu/baseline/07_results.txt # For simulation results

# TARGET=$(ODIR)/bambu/optimized/06_verilog.v
# TARGET=$(ODIR)/bambu/optimized/07_results.txt # For simulation results

# TARGET=$(ODIR)/bambu/transformed/06_verilog.v
TARGET=$(ODIR)/bambu/transformed/07_results.txt # For simulation results


all: $(TARGET)

# Generate MLIR from Torch model
$(ODIR)/01_tosa.mlir: $(FILE_PATH) 
	python $(FILE_PATH) $@ --dialect=tosa

# Put the transform schedule on the output directory where it is expected
$(ODIR)/04_transform_sched.mlir: $(TRANSFORM_PATH) 
	cp $< $@ 

# Include the rules to generate linalg from a torch model translated to linalg
include $(SCRIPTS_DIR)/mkinc/tosa_to_llvm.mk

# Include the rules to generate soda LLVM IR
include $(SCRIPTS_DIR)/mkinc/soda_to_llvm.mk

# Include rules to generate verilog and simulate with Bambu
#  Select the target Bambu device and clock period below, if not specified
#  default values are used.
BAMBU_DEVICE=asap7-BC
# BAMBU_DEVICE=nangate45
BAMBU_CLOCK_PERIOD=5

# IP Integration flow variables
BAMBU_IP_INTEGRATION=true
# Comma-separated lists of files, must have the full path
IP_C_EXCLUDE=$(IPDIR)/sodaInstrAssertLessThen.c
IP_VERILOG_INPUTS=$(IPDIR)/sodaInstrAssertLessThen.v
IP_MODULE_LIB=$(IPDIR)/module_lib.xml
IP_CONSTRAINTS=$(IPDIR)/constraints_STD.xml

# Space-separated list for Makefile dependencies
IP_INTEGRATION_FILES := \
  $(IP_C_EXCLUDE) \
  $(IP_VERILOG_INPUTS) \
  $(IP_MODULE_LIB) \
  $(IP_CONSTRAINTS)
EXTRA_VERILOG_DEPS := $(IP_INTEGRATION_FILES)
include $(SCRIPTS_DIR)/mkinc/llvm_to_verilog.mk
