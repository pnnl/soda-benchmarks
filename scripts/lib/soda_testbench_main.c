/* Test driver for the cpu and esp backends.
 *
 * Everything that varies per kernel lives in the generated testdata.h that
 * torchscript.py --emit-testdata writes: the parameter arrays, their order and
 * count (as the TD_CALL macro, since C cannot spell a variable arity), the
 * output buffer and the golden result PyTorch produced for the same inputs.
 * This file therefore never changes with the kernel.
 *
 * Exit status is always 0. The Makefile rule captures stdout as the artifact,
 * and one of the two supported cpu configurations -- the ESP-lowered kernel
 * against the mock runtime, which never writes the output -- is expected to
 * report a mismatch. `make check` is what turns TEST FAILED into an error.
 */

#include <stdio.h>

#include "testdata.h"

/* esp_prof, declared here rather than via its header so the driver builds
 * identically for the cpu backend (where the mock runtime provides it) and for
 * ESP (where esp_prof.c is staged alongside). Region 0 is ESP_PROF_TOTAL. */
extern void esp_prof_begin(unsigned id);
extern void esp_prof_end(unsigned id);
extern void esp_prof_report(void);

/* Lowered with the bare-pointer memref convention, so every operand -- rank-0
 * scalars included -- arrives as exactly one pointer. TD_DECL spells the
 * prototype because only the generated header knows the arity. */
extern TD_DECL(forward_kernel);

int main(void)
{
    unsigned  i;
    unsigned  errors  = 0;
    td_elem_t max_err = 0;

    printf("kernel=%s dataset=%s dtype=%s elements=%d tol=%g\n", TD_KERNEL,
           TD_DATASET, TD_DTYPE, (int)TD_OUT_N, (double)TD_TOL);

    esp_prof_begin(0);
    TD_CALL(forward_kernel);
    esp_prof_end(0);

    for (i = 0; i < (unsigned)TD_OUT_N; i++) {
        td_elem_t err = TD_OUT[i] - td_golden[i];
        if (err < 0) err = -err;
        if (err > max_err) max_err = err;
        if (err > (td_elem_t)TD_TOL) {
            if (errors < 10)
                printf("  MISMATCH at [%u]: got=%g want=%g err=%g\n", i,
                       (double)TD_OUT[i], (double)td_golden[i], (double)err);
            errors++;
        }
    }

    printf("---------------------------------\n");
    if (errors == 0)
        printf("TEST PASSED (max error: %g)\n", (double)max_err);
    else
        printf("TEST FAILED (%u/%d elements exceed tolerance %g)\n", errors,
               (int)TD_OUT_N, (double)TD_TOL);
    /* ESP's baremetal printf has no float conversions, so %g prints literally
     * there. The same number as a scaled integer, readable on both; capped so a
     * gross failure (the mock runtime, which does not compute) stays legible. */
    {
        double e = (double)max_err * 1e9 + 0.5;
        if (e > 4000000000.0) e = 4000000000.0;
        printf("max error = %u e-9, tolerance = %u e-9\n", (unsigned)e,
               (unsigned)((double)TD_TOL * 1e9 + 0.5));
    }
    printf("---------------------------------\n");

    /* Empty unless the kernel was lowered with profile=true. */
    esp_prof_report();

    return 0;
}
