// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "vectors.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

extern void forward_kernel(float *C, float *A, float *B, float *out,
                           float *beta, float *alpha);
static float c[96], a[128], b[192], out[96], beta, alpha;

static unsigned input_differences(const float *actual, const uint32_t *gold,
                                  unsigned words) {
  unsigned errors = 0;
  for (unsigned i = 0; i < words; i++) {
    uint32_t value;
    memcpy(&value, actual + i, 4);
    errors += (value != gold[i]);
  }
  return errors;
}

int main(void) {
  unsigned total = 0;
  printf("GEMM_SYSTEM BEGIN unchanged_host_and_rtl cases=3\n");
  for (unsigned run = 0; run < 3; run++) {
    memcpy(c, inputs[run], sizeof c);
    memcpy(a, inputs[run] + 96, sizeof a);
    memcpy(b, inputs[run] + 224, sizeof b);
    memcpy(&beta, inputs[run] + 512, 4);
    memcpy(&alpha, inputs[run] + 513, 4);
    memcpy(out, inputs[run] + 416, sizeof out);
    printf("GEMM_SYSTEM START case=%u\n", run);
    forward_kernel(c, a, b, out, &beta, &alpha);
    unsigned errors = 0;
    for (unsigned i = 0; i < 96; i++) {
      uint32_t value;
      memcpy(&value, out + i, 4);
      if (value != expected[run][i]) {
        if (errors < 8)
          printf("GEMM_SYSTEM MISMATCH case=%u index=%u got=%x expected=%x\n",
                 run, i, value, expected[run][i]);
        errors++;
      }
    }
    unsigned input_errors = input_differences(c, inputs[run], 96) +
                            input_differences(a, inputs[run] + 96, 128) +
                            input_differences(b, inputs[run] + 224, 192) +
                            input_differences(&beta, inputs[run] + 512, 1) +
                            input_differences(&alpha, inputs[run] + 513, 1);
    printf("GEMM_SYSTEM RESULT case=%u output_errors=%u input_errors=%u\n", run,
           errors, input_errors);
    total += errors + input_errors;
  }
  printf("GEMM_SYSTEM %s total_errors=%u\n", total ? "FAIL" : "PASS", total);
  return total != 0;
}
