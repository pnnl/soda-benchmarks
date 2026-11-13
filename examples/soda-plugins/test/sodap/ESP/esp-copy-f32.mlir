// RUN: c++ -std=c++17 -O2 -Wall -Wextra -I %S/../../../include \
// RUN:   %S/Inputs/test_esp_copy_f32.cpp \
// RUN:   %S/../../../lib/sodap/ExecutionEngine/EspRuntimeMock.cpp -o %t
// RUN: %t | FileCheck %s
// CHECK: ESP f32 copy PASS

// Exercise the exported C ABI and shared copy implementation without hardware.
