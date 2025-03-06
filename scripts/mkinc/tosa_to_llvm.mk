# This file provides rules to generate LLVM IR from a TOSA MLIR file

$(ODIR)/02_linalg.mlir: $(ODIR)/01_tosa.mlir $(SCRIPTS_DIR)/tosa_to_linalg.sh
	$(SCRIPTS_DIR)/tosa_to_linalg.sh $< $@

$(ODIR)/03_llvm.mlir: $(ODIR)/02_linalg.mlir $(SCRIPTS_DIR)/linalg_to_llvm.sh
	$(SCRIPTS_DIR)/linalg_to_llvm.sh $< $@

$(ODIR)/04_llvm.ll: $(ODIR)/03_llvm.mlir $(SCRIPTS_DIR)/llvm_to_ll.sh
	$(SCRIPTS_DIR)/llvm_to_ll.sh $< $@

# Checks if the first generated file exists, if so, delete all generated files
# This is to avoid deleting files from other directories
PHONY: clean
clean:
	test -f $(ODIR)/01_tosa.mlir && \
	rm -f $(ODIR)/*.mlir && \
	rm -f $(ODIR)/*.ll && \
	rm -r $(ODIR)
	rm -f core
