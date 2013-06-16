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

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

#include "mypaint-utils-stroke-player.h"
#include "mypaint-test-surface.h"
#include "mypaint-test-surface.h"
#include "testutils.h"
#include "mypaint-benchmark.h"

typedef enum {
    SurfaceTransactionPerStrokeTo,
    SurfaceTransactionPerStroke
} SurfaceTransaction;

typedef struct {
    char *test_case_id;
    MyPaintTestsSurfaceFactory factory_function;
    gpointer factory_user_data;

    float brush_size;
    float scale;
    int iterations;
    const char *brush_file;
    SurfaceTransaction surface_transaction;
} SurfaceTestData;

int
test_surface_drawing(void *user_data)
{
    SurfaceTestData *data = (SurfaceTestData *)user_data;

    char * event_data = read_file("events/painting30sec.dat");
    char * brush_data = read_file(data->brush_file);

    assert(event_data);
    assert(brush_data);

    MyPaintSurface *surface = data->factory_function(data->factory_user_data);
    MyPaintBrush *brush = mypaint_brush_new();
    MyPaintUtilsStrokePlayer *player = mypaint_utils_stroke_player_new();

    mypaint_brush_from_string(brush, brush_data);
    mypaint_brush_set_base_value(brush, MYPAINT_BRUSH_SETTING_RADIUS_LOGARITHMIC, log(data->brush_size));

    mypaint_utils_stroke_player_set_brush(player, brush);
    mypaint_utils_stroke_player_set_surface(player, surface);
    mypaint_utils_stroke_player_set_source_data(player, event_data);
    mypaint_utils_stroke_player_set_scale(player, data->scale);

    if (data->surface_transaction == SurfaceTransactionPerStroke) {
        mypaint_utils_stroke_player_set_transactions_on_stroke_to(player, FALSE);
    }

    // Actually run benchmark
    mypaint_benchmark_start(data->test_case_id);
    for (int i=0; i<data->iterations; i++) {
        if (data->surface_transaction == SurfaceTransactionPerStroke) {
            mypaint_surface_begin_atomic(surface);
        }
        mypaint_utils_stroke_player_run_sync(player);

        if (data->surface_transaction == SurfaceTransactionPerStroke) {
            mypaint_surface_end_atomic(surface);
        }
    }
    int result = mypaint_benchmark_end();

    char *png_filename = malloc(snprintf(NULL, 0, "%s.png", data->test_case_id) + 1);
    sprintf(png_filename, "%s.png", data->test_case_id);

    //mypaint_surface_save_png(surface, png_filename, 0, 0, -1, 1);
    // FIXME: check the correctness of the outputted PNG

    free(png_filename);

    mypaint_brush_unref(brush);
    mypaint_surface_unref(surface);
    mypaint_utils_stroke_player_free(player);

    free(event_data);
    free(brush_data);

    return result;
}

char *
create_id(const char *templ, const char *title)
{
    char *id = malloc(snprintf(NULL, 0, templ, title) + 1);
    sprintf(id, templ, title);
    return id;
}

int
mypaint_test_surface_run(int argc, char **argv,
                      MyPaintTestsSurfaceFactory surface_factory,
                      gchar *title, gpointer user_data)
{
    // FIXME: use an environment variable or commandline switch to
    // distinguish between running test as a benchmark (multiple iterations and taking the time)
    // or as a test (just verifying correctness)

    MyPaintTestsSurfaceFactory f = surface_factory;
    gpointer d = user_data;

    SurfaceTestData data[] = {
        {"1", f, d, 2.0, 1.0, 1, "brushes/modelling.myb", SurfaceTransactionPerStrokeTo},
        {"2", f, d, 4.0, 1.0, 1, "brushes/modelling.myb", SurfaceTransactionPerStrokeTo},
        {"3", f, d, 8.0, 1.0, 1, "brushes/modelling.myb", SurfaceTransactionPerStrokeTo},
        {"4", f, d, 16.0, 2.0, 1, "brushes/modelling.myb", SurfaceTransactionPerStrokeTo},
        {"5", f, d, 32.0, 2.0, 1, "brushes/modelling.myb", SurfaceTransactionPerStrokeTo},
        {"6", f, d, 64.0, 2.0, 1, "brushes/modelling.myb", SurfaceTransactionPerStrokeTo},
        {"7", f, d, 128.0, 4.0, 1, "brushes/modelling.myb", SurfaceTransactionPerStrokeTo},
        {"8", f, d, 256.0, 4.0, 1, "brushes/modelling.myb", SurfaceTransactionPerStrokeTo},
        {"9", f, d, 512.0, 4.0, 1, "brushes/modelling.myb", SurfaceTransactionPerStrokeTo},
        {"10", f, d, 2.0, 1.0, 1, "brushes/charcoal.myb", SurfaceTransactionPerStrokeTo},
        {"11", f, d, 4.0, 1.0, 1, "brushes/charcoal.myb", SurfaceTransactionPerStrokeTo},
        {"12", f, d, 8.0, 1.0, 1, "brushes/charcoal.myb", SurfaceTransactionPerStrokeTo},
        {"13", f, d, 16.0, 2.0, 1, "brushes/charcoal.myb", SurfaceTransactionPerStrokeTo},
        {"14", f, d, 32.0, 2.0, 1, "brushes/charcoal.myb", SurfaceTransactionPerStrokeTo},
        {"15", f, d, 64.0, 2.0, 1, "brushes/charcoal.myb", SurfaceTransactionPerStrokeTo},
        {"16", f, d, 128.0, 4.0, 1, "brushes/charcoal.myb", SurfaceTransactionPerStrokeTo},
        {"17", f, d, 256.0, 4.0, 1, "brushes/charcoal.myb", SurfaceTransactionPerStrokeTo},
        {"18", f, d, 512.0, 4.0, 1, "brushes/charcoal.myb", SurfaceTransactionPerStrokeTo},
        {"19", f, d, 2.0, 1.0, 1, "brushes/coarse_bulk_2.myb", SurfaceTransactionPerStrokeTo},
        {"20", f, d, 4.0, 1.0, 1, "brushes/coarse_bulk_2.myb", SurfaceTransactionPerStrokeTo},
        {"21", f, d, 8.0, 1.0, 1, "brushes/coarse_bulk_2.myb", SurfaceTransactionPerStrokeTo},
        {"22", f, d, 16.0, 2.0, 1, "brushes/coarse_bulk_2.myb", SurfaceTransactionPerStrokeTo},
        {"23", f, d, 32.0, 2.0, 1, "brushes/coarse_bulk_2.myb", SurfaceTransactionPerStrokeTo},
        {"24", f, d, 64.0, 2.0, 1, "brushes/coarse_bulk_2.myb", SurfaceTransactionPerStrokeTo},
        {"25", f, d, 128.0, 2.0, 1, "brushes/coarse_bulk_2.myb", SurfaceTransactionPerStrokeTo},
        {"26", f, d, 256.0, 2.0, 1, "brushes/coarse_bulk_2.myb", SurfaceTransactionPerStrokeTo},
//        {"27", f, d, 512.0, 2.0, 1, "brushes/coarse_bulk_2.myb", SurfaceTransactionPerStrokeTo}, // uses to much memory on most machines
        {"28", f, d, 2.0, 1.0, 1, "brushes/bulk.myb", SurfaceTransactionPerStrokeTo},
        {"29", f, d, 4.0, 1.0, 1, "brushes/bulk.myb", SurfaceTransactionPerStrokeTo},
        {"30", f, d, 8.0, 1.0, 1, "brushes/bulk.myb", SurfaceTransactionPerStrokeTo},
        {"31", f, d, 16.0, 2.0, 1, "brushes/bulk.myb", SurfaceTransactionPerStrokeTo},
        {"32", f, d, 32.0, 2.0, 1, "brushes/bulk.myb", SurfaceTransactionPerStrokeTo},
        {"33", f, d, 64.0, 2.0, 1, "brushes/bulk.myb", SurfaceTransactionPerStrokeTo},
        {"34", f, d, 128.0, 4.0, 1, "brushes/bulk.myb", SurfaceTransactionPerStrokeTo},
        {"35", f, d, 256.0, 4.0, 1, "brushes/bulk.myb", SurfaceTransactionPerStrokeTo},
        {"36", f, d, 512.0, 4.0, 1, "brushes/bulk.myb", SurfaceTransactionPerStrokeTo}
    };

    TestCase test_cases[TEST_CASES_NUMBER(data)];
    for (int i = 0; i < TEST_CASES_NUMBER(data); i++) {
        TestCase t;
        t.id = data[i].test_case_id;
        t.function = test_surface_drawing;
        t.user_data = (void *)&data[i];
        test_cases[i] = t;
    };

    int retval = test_cases_run(argc, argv, test_cases, TEST_CASES_NUMBER(test_cases), TEST_CASE_BENCHMARK);

    /*
    for(int i = 0; i < TEST_CASES_NUMBER(test_cases); i++) {
        free(test_cases[i].id);
    }
    */

    return retval;
}
