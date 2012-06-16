
#include <malloc.h>

#include "mypaint-benchmark-surface.h"
#include "mypaint-test-surface.h"
#include "testutils.h"

typedef struct {
    MyPaintTestsSurfaceFactory factory_function;
    gpointer factory_user_data;
} SurfaceBenchmarkData;

int
mypaint_benchmark_surface_run(int argc, char **argv,
                      MyPaintTestsSurfaceFactory surface_factory,
                      gchar *title, gpointer user_data)
{
    SurfaceBenchmarkData data;
    data.factory_function = surface_factory;
    data.factory_user_data = user_data;

    char *test_case_id = malloc(snprintf(NULL, 0, "/benchmark/%s/paint", title) + 1);
    sprintf(test_case_id, "/benchmark/%s/paint", title);

    TestCase test_cases[] = {
       //#{test_case_id, benchmark_surface_drawing, &data},
    };

    int retval = test_cases_run(argc, argv, test_cases, TEST_CASES_NUMBER(test_cases));

    free(test_case_id);

    return retval;
}
