# Compile TensorFlow encoder model to protobuf and MLIR

FILE_PATH=tfscript.py
SCRIPTS_DIR=../../../scripts
ODIR=output

# Targets for protobuf export
PROTOBUF_TARGET=$(ODIR)/model/eels_encoder/saved_model.pb
FROZEN_GRAPH_TARGET=$(ODIR)/frozen_graph.pbtxt
TFMLIR_TARGET=$(ODIR)/tf.mlir
TOSA_TARGET=$(ODIR)/01_tosa.mlir
# TARGET=$(ODIR)/graphs/01_tosa.dot
# TARGET=$(ODIR)/04_llvm.ll
# TARGET=$(ODIR)/bambu/baseline/06_verilog.v
TARGET=$(ODIR)/bambu/baseline/07_results.txt # For simulation results

all: $(TARGET)

# Export encoder model to protobuf and frozen graph
$(PROTOBUF_TARGET) $(FROZEN_GRAPH_TARGET): $(FILE_PATH)
	python $(FILE_PATH)

# Convert frozen graph to MLIR using dockerized tf-mlir-translate
$(TFMLIR_TARGET): $(FROZEN_GRAPH_TARGET)
	$(SCRIPTS_DIR)/check_docker.sh
	$(DOCKER_RUN) tf-mlir-translate \
	  --graphdef-to-mlir \
	  --tf-input-arrays=x1 \
	  --tf-input-data-types=DT_FLOAT \
	  --tf-input-shapes=1,240,1 \
	  --tf-output-arrays=Identity \
	  $< \
	  -o $@

# Convert MLIR to TOSA dialect
$(TOSA_TARGET): $(TFMLIR_TARGET)
	$(SCRIPTS_DIR)/check_docker.sh
	$(DOCKER_RUN) tf-opt \
	  --tf-executor-to-functional-conversion \
	  --tf-region-control-flow-to-functional \
	  --tf-shape-inference \
	  --tf-to-tosa-pipeline \
	  $< \
	  -o $(ODIR)/01_tosa.mlir.tmp
	sed 's/func.func @main(/func.func @forward(/' $(ODIR)/01_tosa.mlir.tmp > $@

# Include the rules to generate linalg from a torch model translated to linalg
include $(SCRIPTS_DIR)/mkinc/tosa_to_llvm.mk

# Include the rules to generate graph from a mlir file
include $(SCRIPTS_DIR)/mkinc/mlir_to_graph.mk

# Include the rules to generate soda LLVM IR
include $(SCRIPTS_DIR)/mkinc/soda_to_llvm.mk

# Include rules to generate verilog and simulate with Bambu
#  Select the target Bambu device and clock period below, if not specified
#  default values are used.
BAMBU_DEVICE=xc7z020-1clg484
BAMBU_MEMPOLICY=ALL_BRAM
# BAMBU_DEVICE=nangate45
# BAMBU_MEMPOLICY=NO_BRAM
BAMBU_CLOCK_PERIOD=10
include $(SCRIPTS_DIR)/mkinc/llvm_to_verilog.mk
