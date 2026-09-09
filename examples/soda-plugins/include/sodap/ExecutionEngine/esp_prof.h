/* Copyright (c) 2011-2026 Columbia University, System Level Design Group */
/* SPDX-License-Identifier: Apache-2.0 */
/*
 * esp_prof — region timing for generated code.
 *
 * Designed to be called from a compiler pass, not by hand. Every entry point
 * takes a single integer and returns nothing, so emitting a call is one
 * `func.call` with one `arith.constant` — no structs, no strings, no floats
 * crossing the boundary.
 *
 *     esp_prof_begin(ESP_PROF_PACK);
 *     ... the marshalling loops the pass generated ...
 *     esp_prof_end(ESP_PROF_PACK);
 *
 * Regions accumulate across invocations, so a kernel that offloads three times
 * (2mm, 3mm, a network layer) reports the total and the count without the pass
 * having to do anything extra.
 *
 * Ids 0..7 are named here so a pass can emit a bare constant and still get a
 * readable report; 8..ESP_PROF_MAX_ID-1 are free for whatever the pass wants to
 * distinguish. Everything prints as integers: ESP's baremetal printf implements
 * %c %d %l %o %p %s %u %x and no float conversions at all.
 */

#ifndef __ESP_PROF_H__
#define __ESP_PROF_H__

#include <stdint.h>

/* <<-- named regions -->> */
#define ESP_PROF_TOTAL    0 /* whole offloaded call                        */
#define ESP_PROF_PACK     1 /* host layout -> accelerator layout           */
#define ESP_PROF_ACCEL    2 /* CMD_START -> STATUS_DONE                    */
#define ESP_PROF_UNPACK   3 /* accelerator layout -> host layout           */
#define ESP_PROF_EPILOGUE 4 /* work the pass left on the CPU (alpha/beta)  */
#define ESP_PROF_ALLOC    5 /* shared buffer + page table                  */
#define ESP_PROF_FLUSH    6 /* cache flush before launch                   */
#define ESP_PROF_USER     8 /* first id a pass may assign freely           */

#define ESP_PROF_MAX_ID 16

#ifdef __cplusplus
extern "C" {
#endif

/* Bracket a region. Calls to the same id accumulate. Nesting is fine as long
 * as ids differ; re-entering an already-open id is ignored, not double-counted. */
void esp_prof_begin(uint32_t id);
void esp_prof_end(uint32_t id);

/* Discard everything measured so far. */
void esp_prof_reset(void);

/* One line per region that was entered at least once: cycles, invocations,
 * mean, and share of ESP_PROF_TOTAL if that region was used. */
void esp_prof_report(void);

/* For a caller that wants the numbers rather than the printout. */
uint64_t esp_prof_cycles(uint32_t id);
uint32_t esp_prof_count(uint32_t id);

/* Optional label for a user id, so the report is readable. Safe to omit. */
void esp_prof_label(uint32_t id, const char *name);

#ifdef __cplusplus
}
#endif

#endif /* __ESP_PROF_H__ */
