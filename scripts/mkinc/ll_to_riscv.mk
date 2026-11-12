# ESP host and accelerator generation. Final linking needs an ESP checkout.
ESP_APP_NAME ?= $(notdir $(patsubst %/,%,$(CURDIR)))
ESP_FLOW ?= baseline
LCS_PROFILE ?= false
LCS_DMA_WORD_STRIDE ?= 2
LCS_OUTSTANDING_LOADS ?= true
LCS_SEQUENTIAL ?= false
LCS_OVERLAP ?= true
LCS_REUSE_BUFFER_BUDGET ?= 262144
LCS_REUSABLE_GEMM ?= false
LCS_GEMM_SCHEDULE ?= sequential
LCS_GEMM_TILE_M ?= 4
LCS_GEMM_TILE_N ?= 4
LCS_GEMM_TILE_K ?= 4
# tosa_to_lcs.sh's default: a rebuilt plugin makes the outputs stale.
MLIR_PLUGIN_PATH ?= $(SCRIPTS_DIR)/../examples/soda-plugins/build/lib/SODAPlugin.so
LCS_TRANSFORM ?= $(SCRIPTS_DIR)/lcs/transform.mlir
# Tile sizes, see scripts/tosa_to_lcs.sh. All empty: the cost model's, with
# the defaults in scripts/lcs/transform.mlir.
LCS_TILE_SIZES ?=
LCS_PLM_BYTES ?= 327680
LCS_DMA_REQUEST_CYCLES ?=
LCS_DMA_WORD_CYCLES ?=
.PHONY: lcs-options cpu-baseline-object
lcs-options:

# Stable timestamps unless common layout/profiling options change.
$(ODIR)/esp/lcs-options: lcs-options
	mkdir -p $(dir $@)
	@printf '%s\n' 'profile=$(LCS_PROFILE)' 'word_stride=$(LCS_DMA_WORD_STRIDE)' 'outstanding_loads=$(LCS_OUTSTANDING_LOADS)' 'sequential=$(LCS_SEQUENTIAL)' 'overlap=$(LCS_OVERLAP)' 'reuse_budget=$(LCS_REUSE_BUFFER_BUDGET)' 'transform=$(LCS_TRANSFORM)' 'tile_sizes=$(LCS_TILE_SIZES)' 'plm_bytes=$(LCS_PLM_BYTES)' 'dma_request_cycles=$(LCS_DMA_REQUEST_CYCLES)' 'dma_word_cycles=$(LCS_DMA_WORD_CYCLES)' 'reusable=$(LCS_REUSABLE_GEMM)' 'gemm_schedule=$(LCS_GEMM_SCHEDULE)' 'gemm_tiles=$(LCS_GEMM_TILE_M),$(LCS_GEMM_TILE_N),$(LCS_GEMM_TILE_K)' > $@.tmp
	@cmp -s $@.tmp $@ && rm $@.tmp || mv $@.tmp $@

LCS_OUTPUTS = $(addprefix $(ODIR)/esp/%/,05_hw_acc.ll 05_kernel_riscv.ll 05_cpu_baseline.ll architecture.xml argument_order.txt host.mlir outlined.mlir layout.txt)
# Multiple targets in one pattern rule share one lowering invocation.
$(LCS_OUTPUTS): $(ODIR)/01_tosa.mlir $(SCRIPTS_DIR)/tosa_to_lcs.sh $(SCRIPTS_DIR)/tosa_to_reusable_gemm.sh $(SCRIPTS_DIR)/lcs/argument_order.py $(SCRIPTS_DIR)/lcs/select_tiles.py $(LCS_TRANSFORM) $(ODIR)/esp/lcs-options $(wildcard $(MLIR_PLUGIN_PATH) $(dir $(MLIR_PLUGIN_PATH))SODATosaPlugin.so)
	LCS_PROFILE=$(LCS_PROFILE) LCS_DMA_WORD_STRIDE=$(LCS_DMA_WORD_STRIDE) LCS_OUTSTANDING_LOADS=$(LCS_OUTSTANDING_LOADS) LCS_SEQUENTIAL=$(LCS_SEQUENTIAL) LCS_OVERLAP=$(LCS_OVERLAP) LCS_REUSE_BUFFER_BUDGET=$(LCS_REUSE_BUFFER_BUDGET) LCS_TRANSFORM=$(LCS_TRANSFORM) LCS_TILE_SIZES='$(LCS_TILE_SIZES)' LCS_PLM_BYTES=$(LCS_PLM_BYTES) LCS_DMA_REQUEST_CYCLES=$(LCS_DMA_REQUEST_CYCLES) LCS_DMA_WORD_CYCLES=$(LCS_DMA_WORD_CYCLES) LCS_REUSABLE_GEMM=$(LCS_REUSABLE_GEMM) LCS_GEMM_SCHEDULE=$(LCS_GEMM_SCHEDULE) LCS_GEMM_TILE_M=$(LCS_GEMM_TILE_M) LCS_GEMM_TILE_N=$(LCS_GEMM_TILE_N) LCS_GEMM_TILE_K=$(LCS_GEMM_TILE_K) $(SCRIPTS_DIR)/tosa_to_lcs.sh $< $(@D)

$(ODIR)/esp/%/testdata.h: $(FILE_PATH) $(ODIR)/esp/%/argument_order.txt
	python $< --emit-testdata $@ --arg-order "$$(cat $(word 2,$^))"

ESP_STAGED_SOURCES = $(SCRIPTS_DIR)/ll_to_riscv.sh $(SCRIPTS_DIR)/lib/soda_testbench_main.c $(wildcard $(SODAP_DIR)/lib/sodap/ExecutionEngine/EspRuntime.cpp $(SODAP_DIR)/lib/sodap/ExecutionEngine/ReusableGemmRuntime.cpp $(SODAP_DIR)/lib/sodap/ExecutionEngine/esp_prof.c $(SODAP_DIR)/include/sodap/ExecutionEngine/ESPRuntime.h $(SODAP_DIR)/include/sodap/ExecutionEngine/ReusableGemmABI.h $(SODAP_DIR)/include/sodap/ExecutionEngine/esp_prof.h)

$(ODIR)/esp/%/06_kernel_riscv.o: $(ODIR)/esp/%/05_kernel_riscv.ll $(ODIR)/esp/%/testdata.h $(ESP_STAGED_SOURCES)
	ESP_REQUIRE_DEVICE=1 ESP_APP_NAME=$(ESP_APP_NAME) TESTDATA_DIR=$(@D) $(SCRIPTS_DIR)/ll_to_riscv.sh $< $@

$(ODIR)/esp/%/cpu-baseline/06_kernel_riscv.o: $(ODIR)/esp/%/05_cpu_baseline.ll $(ODIR)/esp/%/testdata.h $(ESP_STAGED_SOURCES)
	ESP_APP_NAME=$(ESP_APP_NAME)_cpu TESTDATA_DIR=$(dir $<) $(SCRIPTS_DIR)/ll_to_riscv.sh $< $@

cpu-baseline-object: $(ODIR)/esp/$(ESP_FLOW)/cpu-baseline/06_kernel_riscv.o

$(ODIR)/esp/bambu-options: lcs-options
	mkdir -p $(dir $@)
	@printf '%s\n' 'device=$(BAMBU_DEVICE)' 'clock=$(BAMBU_CLOCK_PERIOD)' > $@.tmp
	@cmp -s $@.tmp $@ && rm $@.tmp || mv $@.tmp $@

$(ODIR)/esp/%/06_verilog.v: $(ODIR)/esp/%/05_hw_acc.ll $(ODIR)/esp/%/architecture.xml $(SCRIPTS_DIR)/ll_to_verilog_esp_version.sh $(ODIR)/esp/bambu-options
	BAMBU_DEVICE=$(BAMBU_DEVICE) BAMBU_CLOCK_PERIOD=$(BAMBU_CLOCK_PERIOD) $(SCRIPTS_DIR)/ll_to_verilog_esp_version.sh $< forward_kernel
	cp $(@D)/bambu_syn/forward_kernel.v $@

# Preserve inputs to validation/deployment rather than deleting intermediates.
.PRECIOUS: $(LCS_OUTPUTS) $(ODIR)/esp/%/testdata.h $(ODIR)/esp/%/06_kernel_riscv.o $(ODIR)/esp/%/cpu-baseline/06_kernel_riscv.o

$(ODIR)/esp/%/07_kernel.riscv: $(ODIR)/esp/%/06_kernel_riscv.o
	$(SCRIPTS_DIR)/link_esp_app.sh $< $@
