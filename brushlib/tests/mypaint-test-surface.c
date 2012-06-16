
#include "testutils.h"

#include "mypaint-test-surface.h"

int
mypaint_test_surface_run(int argc, char **argv,
                      MyPaintTestsSurfaceFactory surface_factory,
                      gchar *title, gpointer user_data)
{
    // TODO: add test cases
    TestCase test_cases[] = {
        // {"/test/surface/%s/", test_brush_load_base_values},
    };

    return test_cases_run(argc, argv, test_cases, TEST_CASES_NUMBER(test_cases));
}
