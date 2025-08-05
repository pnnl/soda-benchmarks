//===----------------------------------------------------------------------===//
//
// Part of the SODA Benchmarks Project
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
//
//===----------------------------------------------------------------------===//

#include "module_lib.h"
#include <stdio.h>
#include <stdbool.h>

void sodaInstrHWCounter(uint64_t it, uint64_t max)
{
  bool result = it < max;
  printf("sodaInstrHWCounter: %llu < %llu ? %s\n", it, max, result ? "true" : "false");
}
