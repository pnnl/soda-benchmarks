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
# A pattern rule with several targets is one recipe that makes all of them.
$(ODIR)/esp/%/05_kernel_riscv.ll $(ODIR)/esp/%/05_hw_acc.ll $(ODIR)/esp/%/architecture.xml: $(ODIR)/01_tosa.mlir $(SCRIPTS_DIR)/tosa_to_lcs.sh
	$(SCRIPTS_DIR)/tosa_to_lcs.sh $< $(@D)

# testdata.h stays in $(ODIR), not beside the .ll where ll_to_riscv.sh looks by default.
$(ODIR)/esp/%/06_kernel_riscv.o: $(ODIR)/esp/%/05_kernel_riscv.ll $(ODIR)/testdata.h
	ESP_APP_NAME=$(ESP_APP_NAME) TESTDATA_DIR=$(ODIR) $(SCRIPTS_DIR)/ll_to_riscv.sh $< $@

# ll_to_verilog_esp_version.sh runs Bambu in bambu_syn/ beside its input.
$(ODIR)/esp/%/06_verilog.v: $(ODIR)/esp/%/05_hw_acc.ll $(ODIR)/esp/%/architecture.xml
	$(SCRIPTS_DIR)/ll_to_verilog_esp_version.sh $< forward_kernel
	cp $(@D)/bambu_syn/forward_kernel.v $@

# Keep the object (and its staged esp-app/) when the binary stage chains onto it.
.PRECIOUS: $(ODIR)/esp/%/06_kernel_riscv.o

$(ODIR)/esp/%/07_kernel.riscv: $(ODIR)/esp/%/06_kernel_riscv.o
	$(SCRIPTS_DIR)/link_esp_app.sh $< $@
