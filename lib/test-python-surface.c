
#include <stddef.h>
#include <assert.h>
#include <Python.h>

#include "pythontiledsurface.h"
#include <mypaint-test-surface.h>

int
main(int argc, char **argv)
{
    Py_Initialize();

    int retval = mypaint_test_surface_run(argc, argv,
                                          mypaint_python_surface_factory,
                                          "MyPaintPythonTiledSurface",
                                          NULL);

    Py_Finalize();
    
    return retval;
}

