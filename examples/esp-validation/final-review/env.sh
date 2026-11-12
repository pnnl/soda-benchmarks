#!/usr/bin/env bash
# Source on emilia before running the native simulator or software compiler.
review_tools=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
review_repo=$(cd "$review_tools/../../.." && pwd)
export ESP_REVIEW_ARTIFACTS=${ESP_REVIEW_ARTIFACTS:-$review_repo/build/final-review}
export PATH=/opt/cad/modelsim/bin:/opt/riscv/bin:$PATH
export MODEL_TECH=${MODEL_TECH:-/opt/cad/modelsim/linux_x86_64}
export MGLS_LICENSE_FILE=${MGLS_LICENSE_FILE:-1720@bioeelincad.ee.columbia.edu}
export LM_LICENSE_FILE=${LM_LICENSE_FILE:-$MGLS_LICENSE_FILE}
