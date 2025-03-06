# This file provides rules to generate TOSA from a TFLite model

$(ODIR)/01_tosa.mlir: $(MODEL_PATH) $(SCRIPTS_DIR)/tflite_to_tosa.sh
	$(SCRIPTS_DIR)/tflite_to_tosa.sh $< $@

# Checks if the first generated file exists, if so, delete all generated files
# This is to avoid deleting files from other directories
PHONY: clean
clean:
	test -f $(ODIR)/01_tosa.mlir && \
	rm -f $(ODIR)/*.mlir && \
	rm -r $(ODIR)
	rm -f core
