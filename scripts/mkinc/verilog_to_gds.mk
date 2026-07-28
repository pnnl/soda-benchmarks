# This file provides rules to generate GDS from Verilog using OpenROAD

# Bambu top function name. Also baked into ll_to_verilog.sh and the
# soda_to_llvm_*.sh scripts, so overriding it alone is not yet sufficient.
BAMBU_TOP_FNAME ?= forward_kernel

# OpenROAD PDK directory under results/. BAMBU_DEVICE may carry a corner suffix
# (asap7-BC) that the results directory does not use, so it is stripped. The
# nangate45 fallback covers includers that do not set BAMBU_DEVICE at all.
GDS_PLATFORM ?= $(firstword $(subst -, ,$(or $(BAMBU_DEVICE),nangate45)))

# One rule per transformation flow, all identical apart from the flow directory.
define GDS_RULE
$(ODIR)/bambu/$(1)/HLS_output/Synthesis/bash_flow/openroad/results/$(GDS_PLATFORM)/$(BAMBU_TOP_FNAME)/base/6_final.gds: $(ODIR)/bambu/$(1)/06_verilog.v
	$(SCRIPTS_DIR)/verilog_to_gds.sh $$< $(BAMBU_TOP_FNAME)
endef

$(foreach f,baseline optimized transformed,$(eval $(call GDS_RULE,$(f))))
