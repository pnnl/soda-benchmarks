# This file provides rules to generate Verilog and Simulation results using Bambu

# =============================================================================
# Rules specific to non-optimized baseline
$(ODIR)/bambu/baseline/06_verilog.v: $(ODIR)/05_llvm_baseline.ll
	BAMBU_DEVICE=$(BAMBU_DEVICE) \
	BAMBU_CLOCK_PERIOD=$(BAMBU_CLOCK_PERIOD) \
	BAMBU_MEMPOLICY=$(BAMBU_MEMPOLICY) \
	$(SCRIPTS_DIR)/ll_to_verilog.sh $< $@

$(ODIR)/bambu/baseline/07_results.txt: $(ODIR)/05_llvm_baseline.ll
	BAMBU_RUN_SIMULATION=true \
	BAMBU_DEVICE=$(BAMBU_DEVICE) \
	BAMBU_CLOCK_PERIOD=$(BAMBU_CLOCK_PERIOD) \
	BAMBU_MEMPOLICY=$(BAMBU_MEMPOLICY) \
	$(SCRIPTS_DIR)/ll_to_verilog.sh $< $@

# =============================================================================
# Rules specific to soda-opt optimized	
$(ODIR)/bambu/optimized/06_verilog.v: $(ODIR)/05_llvm_optimized.ll
	BAMBU_DEVICE=$(BAMBU_DEVICE) \
	BAMBU_CLOCK_PERIOD=$(BAMBU_CLOCK_PERIOD) \
	BAMBU_MEMPOLICY=$(BAMBU_MEMPOLICY) \
	$(SCRIPTS_DIR)/ll_to_verilog.sh $< $@
	
$(ODIR)/bambu/optimized/07_results.txt: $(ODIR)/05_llvm_optimized.ll
	BAMBU_RUN_SIMULATION=true \
	BAMBU_DEVICE=$(BAMBU_DEVICE) \
	BAMBU_CLOCK_PERIOD=$(BAMBU_CLOCK_PERIOD) \
	BAMBU_MEMPOLICY=$(BAMBU_MEMPOLICY) \
	$(SCRIPTS_DIR)/ll_to_verilog.sh $< $@
