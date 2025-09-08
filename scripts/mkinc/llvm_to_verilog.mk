# This file provides rules to generate Verilog and Simulation results using Bambu

BAMBU_SETTINGS = \
  BAMBU_DEVICE=$(BAMBU_DEVICE) \
  BAMBU_CLOCK_PERIOD=$(BAMBU_CLOCK_PERIOD) \
  BAMBU_MEMPOLICY=$(BAMBU_MEMPOLICY) \
  BAMBU_IP_INTEGRATION=$(BAMBU_IP_INTEGRATION) \
  IP_C_EXCLUDE=$(IP_C_EXCLUDE) \
  IP_VERILOG_INPUTS=$(IP_VERILOG_INPUTS) \
  IP_MODULE_LIB=$(IP_MODULE_LIB) \
  IP_CONSTRAINTS=$(IP_CONSTRAINTS) \

#  IP_TB=$(IP_TB) \
#  IP_TOP_FNAME=$(IP_TOP_FNAME)

# =============================================================================
# Rules specific to non-optimized baseline
$(ODIR)/bambu/baseline/06_verilog.v: $(ODIR)/05_llvm_baseline.ll
	$(BAMBU_SETTINGS) \
	$(SCRIPTS_DIR)/ll_to_verilog.sh $< $@

$(ODIR)/bambu/baseline/07_results.txt: $(ODIR)/05_llvm_baseline.ll
	BAMBU_RUN_SIMULATION=true \
	$(BAMBU_SETTINGS) \
	$(SCRIPTS_DIR)/ll_to_verilog.sh $< $@

# =============================================================================
# Rules specific to soda-opt optimized	
$(ODIR)/bambu/optimized/06_verilog.v: $(ODIR)/05_llvm_optimized.ll
	$(BAMBU_SETTINGS) \
	$(SCRIPTS_DIR)/ll_to_verilog.sh $< $@
	
$(ODIR)/bambu/optimized/07_results.txt: $(ODIR)/05_llvm_optimized.ll
	BAMBU_RUN_SIMULATION=true \
	$(BAMBU_SETTINGS) \
	$(SCRIPTS_DIR)/ll_to_verilog.sh $< $@

# =============================================================================
# Rules specific to soda-opt transformed
$(ODIR)/bambu/transformed/06_verilog.v: $(ODIR)/05_llvm_transformed.ll $(EXTRA_VERILOG_DEPS)
	$(BAMBU_SETTINGS) \
	$(SCRIPTS_DIR)/ll_to_verilog.sh $< $@
	
$(ODIR)/bambu/transformed/07_results.txt: $(ODIR)/05_llvm_transformed.ll $(EXTRA_VERILOG_DEPS)
	BAMBU_RUN_SIMULATION=true \
	$(BAMBU_SETTINGS) \
	$(SCRIPTS_DIR)/ll_to_verilog.sh $< $@