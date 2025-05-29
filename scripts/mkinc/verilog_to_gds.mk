# This file provides rules to generate Verilog and Simulation results using Bambu

# =============================================================================
# Rules specific to non-optimized baseline
$(ODIR)/bambu/baseline/HLS_output/Synthesis/bash_flow/openroad/results/nangate45/forward_kernel/base/6_final.gds: $(ODIR)/bambu/baseline/06_verilog.v
	$(SCRIPTS_DIR)/verilog_to_gds.sh $<

# =============================================================================
# Rules specific to soda-opt optimized
$(ODIR)/bambu/optimized/HLS_output/Synthesis/bash_flow/openroad/results/nangate45/forward_kernel/base/6_final.gds: $(ODIR)/bambu/optimized/06_verilog.v
	$(SCRIPTS_DIR)/verilog_to_gds.sh $<

# =============================================================================
# Rules specific to mlir transformed with a transform dialect library
$(ODIR)/bambu/transformed/HLS_output/Synthesis/bash_flow/openroad/results/nangate45/forward_kernel/base/6_final.gds: $(ODIR)/bambu/transformed/06_verilog.v
	$(SCRIPTS_DIR)/verilog_to_gds.sh $<