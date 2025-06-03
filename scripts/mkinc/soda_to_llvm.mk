# This file provides rules to generate LLVM IR using SODA-OPT


# =============================================================================
# Rules specific to non-optimized Baseline
$(ODIR)/04_llvm_baseline.mlir: $(ODIR)/02_linalg.mlir $(SCRIPTS_DIR)/soda_to_llvm_baseline.sh
	$(SCRIPTS_DIR)/soda_to_llvm_baseline.sh $< $@
	mv forward_kernel_testbench.c $(ODIR)/forward_kernel_testbench.c

$(ODIR)/05_llvm_baseline.ll: $(ODIR)/04_llvm_baseline.mlir
	mlir-translate $< -o $@ --mlir-to-llvmir

# =============================================================================
# Rules specific to soda-opt optimized
$(ODIR)/04_llvm_optimized.mlir: $(ODIR)/02_linalg.mlir $(SCRIPTS_DIR)/soda_to_llvm_optimized.sh
	$(SCRIPTS_DIR)/soda_to_llvm_optimized.sh $< $@
	mv forward_kernel_testbench.c $(ODIR)/forward_kernel_testbench.c

$(ODIR)/05_llvm_optimized.ll: $(ODIR)/04_llvm_optimized.mlir
	mlir-translate $< -o $@ --mlir-to-llvmir

# =============================================================================
# Rules specific mlir transformed with a transform dialect library
$(ODIR)/04_llvm_transformed.mlir: $(ODIR)/02_linalg.mlir $(ODIR)/04_transform_sched.mlir  $(SCRIPTS_DIR)/soda_to_llvm_transformed.sh
	$(SCRIPTS_DIR)/soda_to_llvm_transformed.sh $< $@ $(ODIR)/04_transform_sched.mlir
	mv forward_kernel_testbench.c $(ODIR)/forward_kernel_testbench.c

$(ODIR)/05_llvm_transformed.ll: $(ODIR)/04_llvm_transformed.mlir
	mlir-translate $< -o $@ --mlir-to-llvmir

# Checks if the first generated file exists, if so, delete all generated files
# This is to avoid deleting files from other directories
# PHONY: clean
# clean:
# 	test -f $(ODIR)/03-01_linalg_searched.mlir && \
# 	rm -f $(ODIR)/*.mlir && \
# 	rm -f $(ODIR)/*.ll && \
# 	rm -r $(ODIR)
# 	rm -f core
