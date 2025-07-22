#===----------------------------------------------------------------------===//
#
# Part of the SODA Benchmarks Project
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
#
#===----------------------------------------------------------------------===//

ROOT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

# Input file
TOP_C := $(ROOT_DIR)/top.c
TOP_LL := $(ROOT_DIR)/top.ll
TOP := $(TOP_LL)

C_EXCLUDE := $(ROOT_DIR)/module1.c,$(ROOT_DIR)/module2.c,$(ROOT_DIR)/printer1.c,$(ROOT_DIR)/printer2.c,$(ROOT_DIR)/sodaInstrAssertLessThen.c
VERILOG_INPUTS := $(ROOT_DIR)/module1.v,$(ROOT_DIR)/module2.v,$(ROOT_DIR)/printer1.v,$(ROOT_DIR)/printer2.v,$(ROOT_DIR)/sodaInstrAssertLessThen.v
MODULE_LIB := $(ROOT_DIR)/module_lib.xml
CONSTRAINTS := $(ROOT_DIR)/constraints_STD.xml
TB := $(ROOT_DIR)/main_test.c

BAMBU_OPTS := --experimental-setup=BAMBU --memory-allocation-policy=ALL_BRAM -O2 -v3 \
	--C-no-parse=$(C_EXCLUDE) \
	--file-input-data=$(VERILOG_INPUTS) \
	--generate-tb=$(TB) --simulate --simulator=VERILATOR --print-dot --generate-components-library

BAMBU := bambu
TIMEOUT := timeout 2h

IP_DIR := IP_integration_hls

.PHONY: all clean

all: $(IP_DIR)/done

$(TOP_LL): $(TOP_C)
	clang -S -emit-llvm $(TOP_C) -o $(TOP_LL)

$(IP_DIR)/done: $(TOP) $(MODULE_LIB) $(CONSTRAINTS) $(TB)
	@rm -rf $(IP_DIR)
	@mkdir -p $(IP_DIR)
	cd $(IP_DIR) && $(TIMEOUT) $(BAMBU) $(TOP) --top-fname=my_ip $(MODULE_LIB) $(CONSTRAINTS) $(BAMBU_OPTS) $(ARGS)
	@touch $(IP_DIR)/done

clean:
	rm -rf $(IP_DIR)
