#ifndef PYTHONTILEDSURFACE_H
#define PYTHONTILEDSURFACE_H

#include <mypaint-surface.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MyPaintPythonTiledSurface MyPaintPythonTiledSurface;

MyPaintPythonTiledSurface *
mypaint_python_tiled_surface_new(PyObject *py_object);

MyPaintSurface *
mypaint_python_surface_factory(gpointer user_data);

#ifdef __cplusplus
}
#endif

#endif // PYTHONTILEDSURFACE_H


