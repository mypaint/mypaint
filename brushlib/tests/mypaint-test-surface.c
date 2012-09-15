
#include <malloc.h>
#include <assert.h>

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
    int iterations;
    float scale;
    const char *brush_file;
    SurfaceTransaction surface_transaction;
} SurfaceTestData;

int
test_surface_drawing(void *user_data)
{
    SurfaceTestData *data = (SurfaceTestData *)user_data;

    const char * event_data = read_file("events/painting30sec.dat");
    const char * brush_data = read_file(data->brush_file);

    assert(event_data);
    assert(brush_data);

    MyPaintSurface *surface = data->factory_function(data->factory_user_data);
    MyPaintBrush *brush = mypaint_brush_new();
    MyPaintUtilsStrokePlayer *player = mypaint_utils_stroke_player_new();

    mypaint_brush_from_string(brush, brush_data);

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

    mypaint_surface_save_png(surface, png_filename, 0, 0, -1, 1);
    // FIXME: check the correctness of the outputted PNG

    free(png_filename);

    mypaint_brush_destroy(brush);
    mypaint_surface_destroy(surface);
    mypaint_utils_stroke_player_free(player);

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

    MyPaintTestsSurfaceFactory factory = surface_factory;
    gpointer data = user_data;

    SurfaceTestData test1_data = {create_id("surface=%s,brush=modelling", title),
                                  factory, data, 1, 4.0,
                                  "brushes/modelling.myb",
                                  SurfaceTransactionPerStrokeTo};
    SurfaceTestData test2_data = {create_id("surface=%s,brush=charcoal", title),
                                  factory, data, 1, 8.0,
                                  "brushes/charcoal.myb",
                                  SurfaceTransactionPerStrokeTo};
    SurfaceTestData test3_data = {create_id("surface=%s,brush=charcoal,transaction=PerStroke", title),
                                  factory, data, 1, 8.0,
                                  "brushes/charcoal.myb",
                                  SurfaceTransactionPerStroke};

    TestCase test_cases[] = {
        {test1_data.test_case_id, test_surface_drawing, &test1_data},
        {test2_data.test_case_id, test_surface_drawing, &test2_data},
        {test3_data.test_case_id, test_surface_drawing, &test3_data},
    };

    int retval = test_cases_run(argc, argv, test_cases, TEST_CASES_NUMBER(test_cases), TEST_CASE_BENCHMARK);

    free(test1_data.test_case_id);
    free(test2_data.test_case_id);
    free(test3_data.test_case_id);

    return retval;
}
