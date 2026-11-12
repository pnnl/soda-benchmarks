#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
exec bash "$here/container.sh" bash -c '
  set -euo pipefail
  cmake -G Ninja -S examples/soda-plugins -B build/plugin \
    -DMLIR_DIR=/opt/llvm-project/lib/cmake/mlir \
    -DLLVM_DIR=/opt/llvm-project/lib/cmake/llvm \
    -DSODAP_BAMBU_ROOT=/opt/panda -DCMAKE_BUILD_TYPE=Release
  cmake --build build/plugin -j4
  python3 -c "from lit.main import main; main()" -v build/plugin/test
  python3 -m unittest discover -s scripts/reusable_nn -v
'
