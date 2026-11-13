#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Source on emilia. Every site-specific setting can be overridden beforehand.
export ESP_HANDOFF_ROOT=${ESP_HANDOFF_ROOT:-$HOME/esp-handoff-20261106}
export ESP_ROOT=${ESP_ROOT:-$ESP_HANDOFF_ROOT/esp}
export ESP_SOC=${ESP_SOC:-$ESP_ROOT/socs/compare}
export ESP_MODELSIM_INI=${ESP_MODELSIM_INI:-$ESP_HANDOFF_ROOT/sim-libraries/modelsim.ini}
export ESP_MODELSIM_WORK=${ESP_MODELSIM_WORK:-$ESP_SOC/modelsim/work}
export MODEL_TECH=${MODEL_TECH:-/opt/cad/modelsim/linux_x86_64}
export PATH=/opt/cad/modelsim/bin:/opt/riscv/bin:$PATH
export MGLS_LICENSE_FILE=${MGLS_LICENSE_FILE:-1720@bioeelincad.ee.columbia.edu}
export LM_LICENSE_FILE=${LM_LICENSE_FILE:-$MGLS_LICENSE_FILE}
export LD_LLD=${LD_LLD:-ld.lld-16}
export ESP_RECOMPILE_SOC_TOP=${ESP_RECOMPILE_SOC_TOP:-1}
export ESP_INSTRUCTION_TRACE=${ESP_INSTRUCTION_TRACE:-0}
export ESP_SIM_RAM_KBYTES=${ESP_SIM_RAM_KBYTES:-16384}
export ESP_SIM_TIME_MS=${ESP_SIM_TIME_MS:-2000}
export LCS_TIMEOUT_CYCLES=${LCS_TIMEOUT_CYCLES:-20000000}
export NN_TIMEOUT_CYCLES=${NN_TIMEOUT_CYCLES:-2000000}
export SODA_VALIDATED=${SODA_VALIDATED:-$ESP_HANDOFF_ROOT/experiments/soda-reuse-lcs-integration-20261107}
