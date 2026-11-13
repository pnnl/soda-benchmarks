#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
source "$here/env.sh"
test -d "$ESP_HANDOFF_ROOT/dependencies/python" || {
  echo "Missing Python dependencies: $ESP_HANDOFF_ROOT/dependencies/python" >&2
  exit 1
}
if (($# == 0)); then set -- bash; fi
interactive=()
if [[ -t 0 && -t 1 ]]; then interactive=(-it); fi
exec docker run --rm "${interactive[@]}" --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$repo,dst=/workspace" \
  --mount "type=bind,src=$ESP_HANDOFF_ROOT/dependencies,dst=/handoff-deps,readonly" \
  -w /workspace "${SODA_IMAGE:-agostini01/soda-trixie:v19.03}" \
  bash -c '
    set -eo pipefail
    source /opt/panda/settings.sh
    export PATH=/workspace/scripts:/opt/soda-opt/bin:/opt/llvm-project/bin:/opt/tensorflow/bin:/opt/torch-mlir/bin:/opt/openroad/OpenROAD/bin:/opt/openroad/yosys/bin:$PATH
    export PYTHONPATH=/handoff-deps/python:/opt/llvm-project/python_packages/mlir_core:/opt/torch-mlir/python_packages/torch_mlir:/opt/soda-opt/python_packages/soda:/workspace
    export TORCH_HOME=/workspace/build/torch-cache
    export MLIR_PLUGIN_PATH=/workspace/build/plugin/lib/SODAPlugin.so
    export SODAP_LIB_DIR=/workspace/build/plugin/lib SODAP_DIR=/workspace/examples/soda-plugins
    export RISCV_CLANG=/usr/bin/clang-19 CLANG=/usr/bin/clang-19 CC=/usr/bin/clang-19
    export LCS_DMA_WORD_STRIDE=2 LCS_PROFILE=true
    exec "$@"
  ' emilia "$@"
