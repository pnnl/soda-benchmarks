# Rules to cross-compile a soda-generated .ll for an ESP SoC's RISC-V core.
# This is the `esp` backend; see docs/ESPBackend.md.
#
# `object` is the deepest stage this container can reach. `binary` needs an ESP
# checkout, and link_esp_app.sh says exactly what to set when it is missing.

# ESP_APP_NAME names the staged baremetal application; default it to the
# experiment directory so two experiments do not collide inside an ESP tree.
ESP_APP_NAME ?= $(notdir $(patsubst %/,%,$(CURDIR)))

# The kernel as a load/compute/store accelerator for Bambu, with its
# architecture.xml, and the host code that drives it.
$(ODIR)/lcs/accelerator.ll $(ODIR)/lcs/host.ll &: $(ODIR)/01_tosa.mlir $(SCRIPTS_DIR)/tosa_to_lcs.sh
	mkdir -p $(ODIR)/lcs
	cp $< $(ODIR)/lcs/01_tosa.mlir
	$(SCRIPTS_DIR)/tosa_to_lcs.sh $(ODIR)/lcs/01_tosa.mlir

$(ODIR)/esp/%/06_kernel_riscv.o: $(ODIR)/05_llvm_%.ll $(ODIR)/testdata.h
	ESP_APP_NAME=$(ESP_APP_NAME) $(SCRIPTS_DIR)/ll_to_riscv.sh $< $@

# Keep the object (and its staged esp-app/) when the binary stage chains onto it.
.PRECIOUS: $(ODIR)/esp/%/06_kernel_riscv.o

$(ODIR)/esp/%/07_kernel.riscv: $(ODIR)/esp/%/06_kernel_riscv.o
	$(SCRIPTS_DIR)/link_esp_app.sh $< $@
