# This file provides rules to generate Verilog and Simulation results using Bambu

# =============================================================================
# Rules specific to non-optimized baseline
$(ODIR)/bambu/baseline/06_verilog.v: $(FILE_PATH)
	mkdir -p $(ODIR)/bambu/baseline
	BAMBU_DEVICE=$(BAMBU_DEVICE) \
	BAMBU_CLOCK_PERIOD=$(BAMBU_CLOCK_PERIOD) \
	BAMBU_MEMPOLICY=$(BAMBU_MEMPOLICY) \
	$(SCRIPTS_DIR)/c_to_verilog.sh $< $@

$(ODIR)/bambu/baseline/07_results.txt: $(FILE_PATH)
	BAMBU_RUN_SIMULATION=true \
	BAMBU_TESTBENCH_FILE=$(SIMULATION_FILE_PATH) \
	BAMBU_DEVICE=$(BAMBU_DEVICE) \
	BAMBU_CLOCK_PERIOD=$(BAMBU_CLOCK_PERIOD) \
	BAMBU_MEMPOLICY=$(BAMBU_MEMPOLICY) \
	$(SCRIPTS_DIR)/c_to_verilog.sh $< $@

# =============================================================================