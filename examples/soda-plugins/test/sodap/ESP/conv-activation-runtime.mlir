// RUN: c++ -std=c++17 -O2 -ffp-contract=off -I %S/../../../include %S/Inputs/test_conv_activation.cpp %S/../../../lib/sodap/ExecutionEngine/ReusableOperatorRuntime.cpp -o %t
// RUN: %t | FileCheck %s
// RUN: not %t mode 2>&1 | FileCheck %s --check-prefix=MODE
// RUN: not %t nan 2>&1 | FileCheck %s --check-prefix=SLOPE
// RUN: not %t negative 2>&1 | FileCheck %s --check-prefix=SLOPE
// RUN: not %t large 2>&1 | FileCheck %s --check-prefix=SLOPE
// CHECK: close socket=0 invocations=4
// CHECK: Convolution activation reuse PASS
// MODE: invalid convolution activation
// SLOPE: invalid LeakyReLU slope
