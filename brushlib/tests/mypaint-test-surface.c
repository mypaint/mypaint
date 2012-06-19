
#include <mypaint-surface.h>
#include <mypaint-brush.h>

#include "testutils.h"
#include "mypaint-test-surface.h"

typedef struct {
    MyPaintTestsSurfaceFactory factory_function;
    MyPaintTestsSurfaceFactory factory_user_data;
} SurfaceTestData;

/* Test that the surface implementation can be drawn to and can save the result
 * currrently just a dont-crash test
 * FIXME: verify the outputted PNG */
int
test_paint_and_save(gpointer user_data)
{
    SurfaceTestData *data = (SurfaceTestData *)user_data;
    const char * brush_data = read_file("brushes/modelling.myb");

    MyPaintSurface *surface = data->factory_function(data->factory_user_data);
    MyPaintBrush *brush = mypaint_brush_new();

    mypaint_brush_from_string(brush, brush_data);

    mypaint_brush_stroke_to(brush, surface, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0);
    mypaint_brush_stroke_to(brush, surface, 10.0, 10.0, 1.0, 0.0, 0.0, 1.0);
    mypaint_brush_stroke_to(brush, surface, 20.0, 20.0, 1.0, 0.0, 0.0, 1.0);

    mypaint_surface_save_png(surface, "output.png", 0, 0, -1, 1);

    mypaint_brush_destroy(brush);
    mypaint_surface_destroy(surface);

    return 1;
}

int
mypaint_test_surface_run(int argc, char **argv,
                      MyPaintTestsSurfaceFactory surface_factory,
                      gchar *title, gpointer user_data)
{
    SurfaceTestData paint_and_save_data;
    paint_and_save_data.factory_function = surface_factory;
    paint_and_save_data.factory_user_data = user_data;

    char *test_case_id = malloc(snprintf(NULL, 0, "/test/surface/%s/paint-and-save", title) + 1);
    sprintf(test_case_id, "/test/surface/%s/paint-and-save", title);

    TestCase test_cases[] = {
        {test_case_id, test_paint_and_save, &paint_and_save_data},
    };

    int retval = test_cases_run(argc, argv, test_cases, TEST_CASES_NUMBER(test_cases));

    free(test_case_id);
    return retval;
}
