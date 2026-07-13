//===----------------------------------------------------------------------===//
//
// Part of the SODA Benchmarks Project
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
//
//===----------------------------------------------------------------------===//

#include "module_lib.h"
#include <stdio.h>
#include <stdint.h>

/// @brief Switches the single active HW counter to `location`.
/// @param location the location to switch to, or the report sentinel
///                 (all bits set, i.e. UINT64_MAX) to print the final
///                 report once.
void sodaInstrChangeLocation(uint64_t location)
{
  if (location == UINT64_MAX) {
    printf("[SW] HW counter FINAL report requested\n");
  } else {
    printf("[SW] HW counter switched to loc: %llu\n",
           (unsigned long long)location);
  }
}
