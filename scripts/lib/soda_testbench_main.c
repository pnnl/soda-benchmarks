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

    TD_CALL(forward_kernel);

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
    printf("---------------------------------\n");

    return 0;
}
