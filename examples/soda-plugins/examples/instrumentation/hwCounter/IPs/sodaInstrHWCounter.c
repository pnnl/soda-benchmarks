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


/// @brief  Decides whether to start or stop the HW counter at location.
/// @param action true to start the counter, false to stop it.
/// @param location 
void sodaInstrHWCounter(bool action, uint64_t location)
{
  if (action) {
    printf("HW counter STARTED at location %llu\n", location);
  } else {
    printf("HW counter STOPPED at location %llu\n", location);
  }
}
