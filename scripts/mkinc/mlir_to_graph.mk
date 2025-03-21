# This file provides rules to generate graphs from an MLIR file

$(ODIR)/graphs/01_tosa.dot: $(ODIR)/01_tosa.mlir $(SCRIPTS_DIR)/mlir_to_graph.sh
	$(SCRIPTS_DIR)/mlir_to_graph.sh $< $@

$(ODIR)/graphs/02_linalg.dot: $(ODIR)/02_linalg.mlir $(SCRIPTS_DIR)/mlir_to_graph.sh
	$(SCRIPTS_DIR)/mlir_to_graph.sh $< $@