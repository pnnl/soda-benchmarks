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
    printf("[SW] HW counter STARTED at loc: %llu\n", location);
  } else {
    printf("[SW] HW counter STOPED at loc: %llu\n", location);
  }
}
