/* This file is part of MyPaint.
 * Copyright (C) 2008-2011 by Martin Renold <martinxyz@gmx.ch>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */

#include "pythontiledsurface.h"

struct MyPaintPythonTiledSurface {
    MyPaintTiledSurface parent;
    PyObject * py_obj;
};

// Forward declare
void free_tiledsurf(MyPaintSurface *surface);

static void
tile_request_start(MyPaintTiledSurface *tiled_surface, MyPaintTileRequest *request)
{
    MyPaintPythonTiledSurface *self = (MyPaintPythonTiledSurface *)tiled_surface;

    const gboolean readonly = request->readonly;
    const int tx = request->tx;
    const int ty = request->ty;
    PyArrayObject* rgba = NULL;

#pragma omp critical
{
    rgba = (PyArrayObject*)PyObject_CallMethod(self->py_obj, "_get_tile_numpy", "(iii)", tx, ty, readonly);
    if (rgba == NULL) {
        request->buffer = NULL;
        printf("Python exception during get_tile_numpy()!\n");
        if (PyErr_Occurred()) {
            PyErr_Print();
        }
    } else {

#ifdef HEAVY_DEBUG
        assert(PyArray_NDIM(rgba) == 3);
        assert(PyArray_DIM(rgba, 0) == tiled_surface->tile_size);
        assert(PyArray_DIM(rgba, 1) == tiled_surface->tile_size);
        assert(PyArray_DIM(rgba, 2) == 4);
        assert(PyArray_ISCARRAY(rgba));
        assert(PyArray_TYPE(rgba) == NPY_UINT16);
#endif
        // tiledsurface.py will keep a reference in its tiledict, at least until the final end_atomic()
        Py_DECREF((PyObject *)rgba);
        request->buffer = (uint16_t*)PyArray_DATA(rgba);
    }
} // #end pragma opt critical


}

static void
tile_request_end(MyPaintTiledSurface *tiled_surface, MyPaintTileRequest *request)
{
    // We modify tiles directly, so don't need to do anything here
}

MyPaintPythonTiledSurface *
mypaint_python_tiled_surface_new(PyObject *py_object)
{
    MyPaintPythonTiledSurface *self = (MyPaintPythonTiledSurface *)malloc(sizeof(MyPaintPythonTiledSurface));

    mypaint_tiled_surface_init(&self->parent, tile_request_start, tile_request_end);
    self->parent.threadsafe_tile_requests = TRUE;

    // MyPaintSurface vfuncs
    self->parent.parent.destroy = free_tiledsurf;

    self->py_obj = py_object; // no need to incref

    return self;
}

void free_tiledsurf(MyPaintSurface *surface)
{
    MyPaintPythonTiledSurface *self = (MyPaintPythonTiledSurface *)surface;
    mypaint_tiled_surface_destroy(&self->parent);
    free(self);
}
