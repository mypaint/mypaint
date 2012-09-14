
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
    const char *test_case_id;
    MyPaintTestsSurfaceFactory factory_function;
    gpointer factory_user_data;
    const char *brush_file;
    SurfaceTransaction surface_transaction;
    int iterations;
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

int
mypaint_test_surface_run(int argc, char **argv,
                      MyPaintTestsSurfaceFactory surface_factory,
                      gchar *title, gpointer user_data)
{
    SurfaceTestData data;
    data.factory_function = surface_factory;
    data.factory_user_data = user_data;
    data.brush_file = "brushes/modelling.myb";
    data.surface_transaction = SurfaceTransactionPerStrokeTo;
    data.iterations = 10;

    // FIXME: use an environment variable or commandline switch to
    // distinguish between running test as a benchmark (multiple iterations and taking the time)
    // or as a test (just verifying correctness)
    char *test_case_id = malloc(snprintf(NULL, 0, "/test/%s/paint", title) + 1);
    sprintf(test_case_id, "/test/%s/paint", title);

    data.test_case_id = test_case_id;

    TestCase test_cases[] = {
        {test_case_id, test_surface_drawing, &data},
    };

    int retval = test_cases_run(argc, argv, test_cases, TEST_CASES_NUMBER(test_cases), TEST_CASE_BENCHMARK);

    free(test_case_id);

    return retval;
}
