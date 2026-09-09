/* Copyright (c) 2011-2026 Columbia University, System Level Design Group */
/* SPDX-License-Identifier: Apache-2.0 */
/* See esp_prof.h. */

#include <stdio.h>
#ifndef __riscv
    #include <time.h>
#endif
#include "esp_prof.h"

/*
 * On the SoC this is the cycle counter. Off it, a monotonic clock in
 * nanoseconds -- so the same library can be exercised natively, against the
 * mock runtime or a plain host build, without a board. The units differ
 * between the two, which is why the report never claims a time: it prints
 * cycles on hardware and relative shares everywhere.
 */
static inline uint64_t esp_prof_now(void)
{
#ifdef __riscv
    uint64_t c;
    asm volatile("csrr %0, mcycle" : "=r"(c));
    return c;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
#endif
}

struct esp_prof_region {
    uint64_t    total;
    uint64_t    open_at;
    uint32_t    count;
    uint32_t    open;
    const char *name;
};

/*
 * The magic guard makes initialisation independent of whether .bss was zeroed
 * at boot -- generated code may call esp_prof_begin() before anything else in
 * the program has run, so this cannot rely on someone else going first.
 */
#define ESP_PROF_MAGIC 0x50524f46ul /* "PROF" */

static struct esp_prof_region g_r[ESP_PROF_MAX_ID];
static unsigned long          g_magic;

static const char *const g_named[] = {"total",    "pack",  "accel", "unpack",
                                      "epilogue", "alloc", "flush", "-"};

static void esp_prof_init_once(void)
{
    unsigned i;

    if (g_magic == ESP_PROF_MAGIC) return;
    for (i = 0; i < ESP_PROF_MAX_ID; i++) {
        g_r[i].total   = 0;
        g_r[i].open_at = 0;
        g_r[i].count   = 0;
        g_r[i].open    = 0;
        g_r[i].name    = (i < sizeof(g_named) / sizeof(g_named[0])) ? g_named[i] : 0;
    }
    g_magic = ESP_PROF_MAGIC;
}

void esp_prof_begin(uint32_t id)
{
    esp_prof_init_once();
    if (id >= ESP_PROF_MAX_ID) return;
    if (g_r[id].open) return; /* already inside: ignore rather than double-count */
    g_r[id].open    = 1;
    g_r[id].open_at = esp_prof_now();
}

void esp_prof_end(uint32_t id)
{
    uint64_t now = esp_prof_now();

    esp_prof_init_once();
    if (id >= ESP_PROF_MAX_ID) return;
    if (!g_r[id].open) return; /* end without begin */
    g_r[id].total += now - g_r[id].open_at;
    g_r[id].count++;
    g_r[id].open = 0;
}

void esp_prof_reset(void)
{
    g_magic = 0;
    esp_prof_init_once();
}

void esp_prof_label(uint32_t id, const char *name)
{
    esp_prof_init_once();
    if (id < ESP_PROF_MAX_ID) g_r[id].name = name;
}

uint64_t esp_prof_cycles(uint32_t id)
{
    esp_prof_init_once();
    return (id < ESP_PROF_MAX_ID) ? g_r[id].total : 0;
}

uint32_t esp_prof_count(uint32_t id)
{
    esp_prof_init_once();
    return (id < ESP_PROF_MAX_ID) ? g_r[id].count : 0;
}

void esp_prof_report(void)
{
    uint64_t denom;
    unsigned i;

    esp_prof_init_once();
    denom = g_r[ESP_PROF_TOTAL].total;

    printf("region        cycles       calls        mean     share\n");
    printf("------------------------------------------------------\n");
    for (i = 0; i < ESP_PROF_MAX_ID; i++) {
        if (g_r[i].count == 0) continue;
        printf("%-12s %11llu %7u %11llu", g_r[i].name ? g_r[i].name : "user",
               (unsigned long long)g_r[i].total, g_r[i].count,
               (unsigned long long)(g_r[i].total / g_r[i].count));
        /* Integer percent: printf here has no float conversions. */
        if (denom) printf(" %7u%%", (unsigned)((g_r[i].total * 100ull) / denom));
        printf("\n");
    }
    printf("------------------------------------------------------\n");
}
