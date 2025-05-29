# Compile Torch model to LLVM

FILE_PATH=torchscript.py
SCRIPTS_DIR=../../../scripts
# Make sure the output directory is not empty, many files will be generated.
# ODIR=.
ODIR=output

# Edit to change the target
# TARGET=$(ODIR)/04_llvm.ll
TARGET=$(ODIR)/bambu/06_verilog.v

all: $(TARGET)

# MLIR file is generated with the torch_mlir python libraries. The file can be
# generated at the tosa dialect level. However, empirical observations show that
# generating the model at the linalg dialect level yields less instructions in
# the final LLVM IR.  
$(ODIR)/01_tosa.mlir: $(FILE_PATH) 
	python $(FILE_PATH) $@ --dialect=tosa

# Include the rules to generate linalg from a torch model translated to linalg
include $(SCRIPTS_DIR)/mkinc/tosa_to_llvm.mk

# Include the rules to generate soda LLVM IR
include $(SCRIPTS_DIR)/mkinc/soda_to_llvm.mk

$(ODIR)/bambu/06_verilog.v $(ODIR)/bambu/forward_kernel.v $(ODIR)/bambu/results.txt: $(ODIR)/05_llvm_baseline.ll
	mkdir -p $(ODIR)/bambu
	# TODO: test opt optimizations
	# opt -O2 -S -o $(ODIR)/bambu/input.ll $< --disable-loop-unrolling
	cp $< $(ODIR)/bambu/input.ll

	cd $(ODIR)/bambu && \
	bambu -v3 --print-dot \
		-lm --soft-float \
		--compiler=I386_CLANG16  \
		--device=asap7-BC \
		--clock-period=5 \
		--experimental-setup=BAMBU-BALANCED-MP \
		--channels-number=2 \
		--memory-allocation-policy=ALL_BRAM \
		--disable-function-proxy \
		--generate-tb=../forward_kernel_testbench.c \
		--simulate --simulator=VERILATOR \
		--verilator-parallel \
		--top-fname=forward_kernel \
		input.ll 2>&1 | tee bambu-log

	cp $(ODIR)/bambu/forward_kernel.v $(ODIR)/bambu/06_verilog.v
