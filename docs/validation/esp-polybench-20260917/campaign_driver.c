/* Campaign driver: all output tensors, finite checks, unchanged TD_TOL.
 * Exit status stays zero for compatibility with the upstream mock flow;
 * the experiment runner must inspect TEST PASSED / TEST FAILED.
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "testdata.h"
#ifndef TD_MODE
#define TD_MODE native
#endif
#define TD_STRINGIFY_INNER(x) #x
#define TD_STRINGIFY(x) TD_STRINGIFY_INNER(x)

extern void esp_prof_begin(unsigned id);
extern void esp_prof_end(unsigned id);
extern void esp_prof_report(void);
extern TD_DECL(forward_kernel);

/* Support legacy, single-output testdata.h too. */
#ifndef TD_NOUTPUTS
#define TD_NOUTPUTS 1
static td_elem_t *const td_outputs[] = {TD_OUT};
static const td_elem_t *const td_goldens[] = {td_golden};
static const unsigned td_sizes[] = {TD_OUT_N};
#endif

/* ESP uses -ffast-math: isfinite()/isnan() are not reliable there.
 * Classify IEEE-754 exponent bits without floating-point predicates. */
static int finite_elem(const td_elem_t *value)
{
    if (sizeof(td_elem_t) == sizeof(uint32_t)) {
        uint32_t bits;
        memcpy(&bits, value, sizeof(bits));
        return (bits & UINT32_C(0x7f800000)) != UINT32_C(0x7f800000);
    } else {
        uint64_t bits;
        memcpy(&bits, value, sizeof(bits));
        return (bits & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
    }
}

static unsigned scaled_error(td_elem_t value)
{
    double scaled = (double)value * 1e9 + 0.5;
    if (scaled > 4000000000.0) scaled = 4000000000.0;
    return (unsigned)scaled;
}

int main(void)
{
    unsigned output, i, elements = 0, errors = 0, nonfinite = 0;
    td_elem_t max_err = 0;

    for (output = 0; output < TD_NOUTPUTS; ++output) {
        elements += td_sizes[output];
        /* All-ones is a NaN for float and double; catches unwritten results.
         * Setup and reference checking are outside the timed region. */
        memset(td_outputs[output], 0xff, td_sizes[output] * sizeof(td_elem_t));
    }
    printf("kernel=%s mode=%s dataset=%s dtype=%s outputs=%u elements=%u\n",
           TD_KERNEL, TD_STRINGIFY(TD_MODE), TD_DATASET, TD_DTYPE,
           (unsigned)TD_NOUTPUTS, elements);
    esp_prof_begin(0);
    TD_CALL(forward_kernel);
    esp_prof_end(0);

    for (output = 0; output < TD_NOUTPUTS; ++output) {
        for (i = 0; i < td_sizes[output]; ++i) {
            const td_elem_t *got = &td_outputs[output][i];
            const td_elem_t *want = &td_goldens[output][i];
            if (!finite_elem(got) || !finite_elem(want)) {
                if (errors < 10)
                    printf("  NONFINITE output=%u index=%u\n", output, i);
                ++nonfinite;
                ++errors;
                continue;
            }
            td_elem_t err = *got - *want;
            if (err < 0) err = -err;
            if (!finite_elem(&err)) {
                ++nonfinite;
                ++errors;
                continue;
            }
            if (err > max_err) max_err = err;
            if (err > (td_elem_t)TD_TOL) {
                if (errors < 10)
                    printf("  MISMATCH output=%u index=%u error=%u e-9\n",
                           output, i, scaled_error(err));
                ++errors;
            }
        }
    }
    printf("---------------------------------\n");
    printf("TEST %s (errors=%u/%u, nonfinite=%u)\n",
           errors ? "FAILED" : "PASSED", errors, elements, nonfinite);
    /* The finite maximum is capped at 4.0 for integer-only UART printing.
     * Nonfinite failures are counted separately, never accepted as zero error. */
    printf("max error = %u e-9, tolerance = %u e-9\n",
           scaled_error(max_err), scaled_error((td_elem_t)TD_TOL));
    printf("---------------------------------\n");
    esp_prof_report();
    return 0;
}
