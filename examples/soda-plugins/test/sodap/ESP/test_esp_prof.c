/* Copyright (c) 2011-2026 Columbia University, System Level Design Group */
/* SPDX-License-Identifier: Apache-2.0 */
/*
 * Native self-test for esp_prof. Builds and runs on a workstation:
 *
 *     cc -O2 -o test_esp_prof test_esp_prof.c esp_prof.c && ./test_esp_prof
 *
 * Exercises the behaviour a compiler pass would depend on, without a board:
 * accumulation across invocations, nesting, unmatched begin/end, out-of-range
 * ids, and the report. Every check is asserted, so a regression fails loudly
 * rather than printing something plausible.
 */

#include <stdio.h>
#include <time.h>
#include "esp_prof.h"

static int failures;

static void check(const char *what, int ok)
{
    printf("  %-58s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) failures++;
}

/* Burn a measurable amount of wall time without relying on sleep. */
static void spin(long ns)
{
    struct timespec a, b;
    clock_gettime(CLOCK_MONOTONIC, &a);
    do {
        clock_gettime(CLOCK_MONOTONIC, &b);
    } while ((b.tv_sec - a.tv_sec) * 1000000000L + (b.tv_nsec - a.tv_nsec) < ns);
}

int main(void)
{
    printf("esp_prof self-test\n\n");

    /* --- accumulation across invocations ------------------------------- */
    /* This is what makes the pass's job easy: it can bracket a region that
     * happens to sit inside a loop without knowing the trip count. */
    esp_prof_reset();
    for (int i = 0; i < 3; i++) {
        esp_prof_begin(ESP_PROF_PACK);
        spin(1000000); /* 1 ms */
        esp_prof_end(ESP_PROF_PACK);
    }
    check("three invocations accumulate into one region", esp_prof_count(ESP_PROF_PACK) == 3);
    check("accumulated total is the sum, not the last",
          esp_prof_cycles(ESP_PROF_PACK) > 2500000ull);

    /* --- nesting -------------------------------------------------------- */
    /* TOTAL wraps the whole call; ACCEL sits inside it. Distinct ids, so both
     * must record independently and TOTAL must be the larger. */
    esp_prof_reset();
    esp_prof_begin(ESP_PROF_TOTAL);
    spin(500000);
    esp_prof_begin(ESP_PROF_ACCEL);
    spin(2000000);
    esp_prof_end(ESP_PROF_ACCEL);
    spin(500000);
    esp_prof_end(ESP_PROF_TOTAL);
    check("nested region records independently", esp_prof_count(ESP_PROF_ACCEL) == 1);
    check("enclosing region is larger than the nested one",
          esp_prof_cycles(ESP_PROF_TOTAL) > esp_prof_cycles(ESP_PROF_ACCEL));

    /* --- misuse must not corrupt --------------------------------------- */
    /* A pass can emit an unbalanced pair through a branch; that must lose the
     * measurement, not the accumulated state. */
    esp_prof_reset();
    esp_prof_begin(ESP_PROF_UNPACK);
    spin(1000000);
    esp_prof_end(ESP_PROF_UNPACK);
    uint64_t before = esp_prof_cycles(ESP_PROF_UNPACK);
    esp_prof_end(ESP_PROF_UNPACK); /* end without begin */
    check("end without begin is ignored", esp_prof_cycles(ESP_PROF_UNPACK) == before &&
                                              esp_prof_count(ESP_PROF_UNPACK) == 1);

    esp_prof_begin(ESP_PROF_UNPACK);
    esp_prof_begin(ESP_PROF_UNPACK); /* re-enter an open region */
    esp_prof_end(ESP_PROF_UNPACK);
    check("re-entering an open region does not double-count",
          esp_prof_count(ESP_PROF_UNPACK) == 2);

    esp_prof_begin(ESP_PROF_MAX_ID + 5); /* out of range */
    esp_prof_end(ESP_PROF_MAX_ID + 5);
    check("out-of-range id is ignored", esp_prof_cycles(ESP_PROF_MAX_ID + 5) == 0);

    /* --- user ids and labels ------------------------------------------- */
    esp_prof_reset();
    esp_prof_label(ESP_PROF_USER, "my-region");
    esp_prof_begin(ESP_PROF_USER);
    spin(1000000);
    esp_prof_end(ESP_PROF_USER);
    check("user id records", esp_prof_count(ESP_PROF_USER) == 1);

    /* --- the report ----------------------------------------------------- */
    /* Shape it the way a real offload looks, so the printed output is
     * representative rather than synthetic. */
    esp_prof_reset();
    esp_prof_begin(ESP_PROF_TOTAL);
    esp_prof_begin(ESP_PROF_ALLOC);  spin(300000);   esp_prof_end(ESP_PROF_ALLOC);
    for (int i = 0; i < 2; i++) {
        esp_prof_begin(ESP_PROF_PACK);   spin(4000000); esp_prof_end(ESP_PROF_PACK);
        esp_prof_begin(ESP_PROF_ACCEL);  spin(3000000); esp_prof_end(ESP_PROF_ACCEL);
        esp_prof_begin(ESP_PROF_UNPACK); spin(1500000); esp_prof_end(ESP_PROF_UNPACK);
    }
    esp_prof_end(ESP_PROF_TOTAL);

    printf("\nrepresentative report (two invocations):\n\n");
    esp_prof_report();

    printf("\n%s  (%d failure%s)\n", failures ? "FAILED" : "PASSED", failures,
           failures == 1 ? "" : "s");
    return failures != 0;
}
