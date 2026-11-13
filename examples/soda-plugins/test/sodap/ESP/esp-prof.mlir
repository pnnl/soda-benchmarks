// RUN: cc -O2 -Wall -Wextra -I %S/../../../include/sodap/ExecutionEngine \
// RUN:   %S/test_esp_prof.c %S/../../../lib/sodap/ExecutionEngine/esp_prof.c -o %t
// RUN: %t

// The .mlir wrapper makes the native C self-test discoverable by this lit suite.
