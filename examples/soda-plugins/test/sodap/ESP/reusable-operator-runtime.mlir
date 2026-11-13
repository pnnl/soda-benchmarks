// RUN: c++ -std=c++17 -O2 -ffp-contract=off -I %S/../../../include \
// RUN:   %S/Inputs/test_reusable_operator.cpp \
// RUN:   %S/../../../lib/sodap/ExecutionEngine/ReusableOperatorRuntime.cpp -o %t
// RUN: %t | FileCheck %s
// RUN: c++ -std=c++17 -O2 -ffp-contract=off -DSODAP_OP_CPU_ONLY -DSODAP_OP_CPU_ARENA_BYTES=1024 \
// RUN:   -I %S/../../../include %S/Inputs/test_reusable_operator.cpp \
// RUN:   %S/../../../lib/sodap/ExecutionEngine/ReusableOperatorRuntime.cpp -o %t.cpu
// RUN: %t.cpu | FileCheck %s
// RUN: not %t handle 2>&1 | FileCheck %s --check-prefix=HANDLE
// RUN: not %t bounds 2>&1 | FileCheck %s --check-prefix=BOUNDS
// RUN: not %t overlap 2>&1 | FileCheck %s --check-prefix=OVERLAP
// RUN: not %t alignment 2>&1 | FileCheck %s --check-prefix=ALIGNMENT
// RUN: not %t duplicate 2>&1 | FileCheck %s --check-prefix=DUPLICATE
// RUN: not %t flag 2>&1 | FileCheck %s --check-prefix=FLAG
// RUN: not %t permutation 2>&1 | FileCheck %s --check-prefix=PERMUTATION
// CHECK: close socket=0 invocations=2
// CHECK: Reusable operator runtime PASS
// HANDLE: invalid context handle
// BOUNDS: invalid or oversized tensor dimensions
// OVERLAP: output overlaps an input
// ALIGNMENT: tensor exceeds arena or is misaligned
// DUPLICATE: socket already owned
// FLAG: invalid boolean parameter
// PERMUTATION: invalid permutation
