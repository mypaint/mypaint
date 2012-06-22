
#include "mypaint-benchmark.h"
#include <mypaint-glib-compat.h>

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <limits.h>

#ifdef HAVE_GPERFTOOLS
#include <profiler.h>
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
        fprintf(stderr, "Warning: Not built with gperftools support.");
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

    fprintf(stderr, "time spent: %f\n", time_spent);

    if (profiling_enabled()) {
#ifdef HAVE_GPERFTOOLS
        ProfileStop();
#else
        fprintf(stderr, "Warning: Not built with gperftools support.");
#endif
    }

    assert(time_spent*1000 < INT_MAX);
    return (int)(time_spent*1000);
}
