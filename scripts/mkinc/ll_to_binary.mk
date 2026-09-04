# Rules to link a soda-generated .ll into a native executable and run it.
# This is the `cpu` backend; see docs/ESPBackend.md.
#
# Pattern rules on the flow name rather than one block per flow: unlike the Bambu
# rules these take no per-flow settings, so `%` is the whole difference.

# The inputs and the PyTorch golden the driver checks against. Needs a benchmark,
# so an experiment scaffolded without --benchmark_name cannot use this backend.
# The argument order comes from the outliner rather than from @forward's
# signature -- see scripts/kernel_arg_order.sh.
$(ODIR)/testdata.h: $(FILE_PATH) $(ODIR)/02_linalg.mlir
	python $< --emit-testdata $@ \
	  --arg-order "$$($(SCRIPTS_DIR)/kernel_arg_order.sh $(ODIR)/02_linalg.mlir)"

$(ODIR)/cpu/%/06_kernel: $(ODIR)/05_llvm_%.ll $(ODIR)/testdata.h
	$(SCRIPTS_DIR)/ll_to_binary.sh $< $@

# Without this make treats the executable as an intermediate of the chained
# pattern rules and deletes it after producing the results file.
.PRECIOUS: $(ODIR)/cpu/%/06_kernel

$(ODIR)/cpu/%/07_results.txt: $(ODIR)/cpu/%/06_kernel
	cd $(dir $@) && ./$(notdir $<) 2>&1 | tee $(notdir $@)

# The driver always exits 0 so a by-design mismatch (an ESP-lowered kernel run
# against the mock runtime, which never computes) still produces its artifact.
# This is the target that turns TEST FAILED into a non-zero exit, so it only
# means anything when TARGET is a cpu backend results file.
.PHONY: check
check: $(TARGET)
	@grep -q 'TEST PASSED' $(TARGET) 2>/dev/null \
	  || { echo "check: $(TARGET) does not report TEST PASSED"; \
	       echo "  (check reads a cpu backend 07_results.txt; TARGET is"; \
	       echo "   $(TARGET))"; exit 1; }
	@echo "check: $(TARGET) reports TEST PASSED"
