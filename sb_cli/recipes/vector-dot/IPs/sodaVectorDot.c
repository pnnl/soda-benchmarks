//===----------------------------------------------------------------------===//
//
// Part of the SODA Benchmarks Project
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
//
//===----------------------------------------------------------------------===//

#include "module_lib.h"
#include <stdint.h>

/// @brief Software emulation of the sodaVectorDot HW module: computes
///        out[0] = sum(a[i] * b[i]) for i in [0, len).
/// @param a first input vector (len elements)
/// @param b second input vector (len elements)
/// @param len number of elements in `a`/`b`
/// @param out single-element output buffer receiving the reduced result
void sodaVectorDot(const float *a, const float *b, uint64_t len, float *out)
{
  float acc = 0.0f;
  for (uint64_t i = 0; i < len; ++i) {
    acc += a[i] * b[i];
  }
  *out = acc;
}
