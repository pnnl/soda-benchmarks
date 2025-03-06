#!/bin/bash

set -e -o pipefail

# Using this template make input and output arguments from the command line:

tf-mlir-translate --graphdef-to-mlir --tf-output-arrays=Identity \
  --tf-input-arrays=Placeholder --tf-input-shapes=1,256,256,1 \
  --tf-enable-shape-inference-on-import \
  $1 | tf-opt --tf-executor-to-functional-conversion \
  --tf-to-tosa-pipeline -o $2
