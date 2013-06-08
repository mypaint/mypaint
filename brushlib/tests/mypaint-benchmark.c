/* brushlib - The MyPaint Brush Library
 * Copyright (C) 2012 Jon Nordby <jononor@gmail.com>
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#include "mypaint-benchmark.h"
#include <mypaint-glib-compat.h>

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <limits.h>

#ifdef HAVE_GPERFTOOLS
#include <gperftools/profiler.h>
#endif

#ifdef WIN32
#include <windows.h>
double get_time()
{
    LARGE_INTEGER t, f;
    QueryPerformanceCounter(&t);
    QueryPerformanceFrequency(&f);
    return (double)t.QuadPart/(double)f.QuadPart;
}

#else
#include <sys/time.h>
#include <sys/resource.h>
double get_time()
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec*1e-6;
}
#endif

static double g_start_time = 0.0;

gboolean
profiling_enabled()
{
    char * enabled = getenv("MYPAINT_ENABLE_PROFILING");
    if (enabled != NULL && strcmp(enabled, "1") == 0) {
        return TRUE;
    }
    return FALSE;
}

void mypaint_benchmark_start(const char *name)
{
    if (profiling_enabled()) {
#ifdef HAVE_GPERFTOOLS
        ProfilerStart(name);
#else
        fprintf(stderr, "Warning: Not built with gperftools support.\n");
#endif
    }

    g_start_time = get_time();
}

/**
 * returns number of milliseconds spent since _start()
 */
int mypaint_benchmark_end()
{
    double time_spent = get_time() - g_start_time;
    g_start_time = 0.0;

    if (profiling_enabled()) {
#ifdef HAVE_GPERFTOOLS
        ProfilerStop();
#else
        fprintf(stderr, "Warning: Not built with gperftools support.\n");
#endif
    }

    assert(time_spent*1000 < INT_MAX);
    return (int)(time_spent*1000);
}
