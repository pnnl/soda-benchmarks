# Compile Torch model to LLVM

FILE_PATH=torchscript.py
SCRIPTS_DIR=../../../scripts
# Make sure the output directory is not empty, many files will be generated.
# ODIR=.
ODIR=output

# Edit to change the target
# TARGET=$(ODIR)/04_llvm.ll
TARGET=$(ODIR)/bambu/05_verilog.v

all: $(TARGET)

# MLIR file is generated with the torch_mlir python libraries. The file can be
# generated at the tosa dialect level. However, empirical observations show that
# generating the model at the linalg dialect level yields less instructions in
# the final LLVM IR.  
$(ODIR)/01_tosa.mlir: $(FILE_PATH) 
	python $(FILE_PATH) $@ --dialect=tosa

# Include the rules to generate linalg from a torch model translated to linalg
include $(SCRIPTS_DIR)/mkinc/tosa_to_llvm.mk

# Include the rules to generate graph from a mlir file
# include $(SCRIPTS_DIR)/mkinc/llvm_to_verilog.mk

# $(SCRIPTS_DIR)/llvm_to_verilog.sh $(ODIR)/04_llvm.ll $(ODIR)/05_verilog.v

$(ODIR)/bambu/05_verilog.v: $(ODIR)/04_llvm.ll
	mkdir -p $(ODIR)/bambu
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
		--top-fname=forward \
		input.ll 2>&1 | tee bambu-log

	cp $(ODIR)/bambu/forward.v $(ODIR)/bambu/05_verilog.v