// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "sodap/ExecutionEngine/ReusableOperatorABI.h"
#include <cstdio>
int main() {
  std::printf("localparam [31:0] OP_CONV_MAGIC=32'h%08x;\n", OP_CONV_MAGIC);
  std::printf("localparam [31:0] OP_ADD_MAGIC=32'h%08x;\n", OP_ADD_MAGIC);
  std::printf("localparam [31:0] GEMM_MAGIC=32'h%08x;\n", GEMM_MAGIC);
}
