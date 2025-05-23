#/bin/bash

set -e -o pipefail

# Copy this script to your project directory and edit the variables below:

MODEL_PATH=<path to tflite model.tflite>
SCRIPTS_DIR=<path to soda scripts directory>
# Make sure the output directory is not empty, many files will be generated.
ODIR=output

$SCRIPTS_DIR/tflite_to_tosa.sh $MODEL_PATH $ODIR/01_tosa.mlir
$SCRIPTS_DIR/tosa_to_linalg.sh $ODIR/01_tosa.mlir $ODIR/02_linalg.mlir
$SCRIPTS_DIR/linalg_to_llvm.sh $ODIR/02_linalg.mlir $ODIR/03_llvm.mlir
$SCRIPTS_DIR/llvm_to_ll.sh     $ODIR/03_llvm.mlir $ODIR/04_llvm.ll
