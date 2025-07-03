# Compile TensorFlow encoder model to protobuf and MLIR

FILE_PATH=tfscript.py
SCRIPTS_DIR=../../../scripts
ODIR=output

# Targets for protobuf export
PROTOBUF_TARGET=$(ODIR)/model/eels_encoder/saved_model.pb
FROZEN_GRAPH_TARGET=$(ODIR)/frozen_graph.pbtxt
MLIR_TARGET=$(ODIR)/tf.mlir
TOSA_TARGET=$(ODIR)/01_tosa.mlir
# TARGET=$(ODIR)/graphs/01_tosa.dot
# TARGET=$(ODIR)/04_llvm.ll
TARGET=$(ODIR)/bambu/baseline/06_verilog.v

all: $(PROTOBUF_TARGET) $(FROZEN_GRAPH_TARGET) $(TOSA_TARGET) $(TARGET)

# Export encoder model to protobuf and frozen graph
$(PROTOBUF_TARGET) $(FROZEN_GRAPH_TARGET): $(FILE_PATH)
	python $(FILE_PATH)

# Convert frozen graph to MLIR using dockerized tf-mlir-translate
$(MLIR_TARGET): $(FROZEN_GRAPH_TARGET)
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
$(TOSA_TARGET): $(MLIR_TARGET)
	$(SCRIPTS_DIR)/check_docker.sh
	$(DOCKER_RUN) tf-opt \
	  --tf-executor-to-functional-conversion \
	  --tf-region-control-flow-to-functional \
	  --tf-shape-inference \
	  --tf-to-tosa-pipeline \
	  $< \
	  -o $@

# Include the rules to generate linalg from a torch model translated to linalg
include $(SCRIPTS_DIR)/mkinc/tosa_to_llvm.mk

# Include the rules to generate graph from a mlir file
include $(SCRIPTS_DIR)/mkinc/mlir_to_graph.mk

# Since soda-opt is not modifying the LLVM IR (yet) we need to update the function name
$(ODIR)/05_llvm_baseline.ll: $(ODIR)/04_llvm.ll
	sed 's/define void @main(/define void @forward_kernel(/' $< > $@

# Include rules to generate verilog and simulate with Bambu
#  Select the target Bambu device and clock period below, if not specified
#  default values are used.
BAMBU_DEVICE=xc7z020-1clg484
BAMBU_CLOCK_PERIOD=10
include $(SCRIPTS_DIR)/mkinc/llvm_to_verilog.mk

