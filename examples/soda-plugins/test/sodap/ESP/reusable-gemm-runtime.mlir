// RUN: c++ -std=c++17 -O2 -ffp-contract=off -I %S/../../../include \
// RUN:   %S/Inputs/test_reusable_gemm.cpp \
// RUN:   %S/../../../lib/sodap/ExecutionEngine/ReusableGemmRuntime.cpp -o %t
// RUN: %t | FileCheck %s
// RUN: not %t handle 2>&1 | FileCheck %s --check-prefix=HANDLE
// RUN: not %t bounds 2>&1 | FileCheck %s --check-prefix=BOUNDS
// RUN: not %t overlap 2>&1 | FileCheck %s --check-prefix=OVERLAP
// RUN: not %t alignment 2>&1 | FileCheck %s --check-prefix=BOUNDS
// RUN: not %t duplicate 2>&1 | FileCheck %s --check-prefix=DUPLICATE
// CHECK: close instance=0 invocations=10
// CHECK: Reusable GEMM runtime PASS
// HANDLE: invalid context handle
// BOUNDS: invalid matrix descriptor
// OVERLAP: output overlaps an input
// DUPLICATE: device instance already owned
