/* This file is part of MyPaint.
 * Copyright (C) 2008-2011 by Martin Renold <martinxyz@gmx.ch>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */

#include "pythontiledsurface.h"

struct _MyPaintPythonTiledSurface {
    MyPaintTiledSurface parent;
    PyObject * py_obj;
    int atomic;
};

// Forward declare
void free_tiledsurf(MyPaintSurface *surface);

void begin_atomic(MyPaintSurface *surface)
{
    MyPaintPythonTiledSurface *self = (MyPaintPythonTiledSurface *)surface;

    mypaint_tiled_surface_begin_atomic((MyPaintTiledSurface *)self);

    self->atomic++;
}

MyPaintRectangle *end_atomic(MyPaintSurface *surface)
{
    MyPaintPythonTiledSurface *self = (MyPaintPythonTiledSurface *)surface;

    MyPaintRectangle *bbox = mypaint_tiled_surface_end_atomic((MyPaintTiledSurface *)self);

    assert(self->atomic > 0);
    self->atomic--;

    if (self->atomic == 0) {
        if (bbox->width > 0) {
            PyObject* res;
            res = PyObject_CallMethod(self->py_obj, "notify_observers", "(iiii)",
                                      bbox->x, bbox->y, bbox->width, bbox->height);
            Py_DECREF(res);
        }
    }

    return bbox;
}

static void
tile_request_start(MyPaintTiledSurface *tiled_surface, MyPaintTiledSurfaceTileRequestData *request)
{
    MyPaintPythonTiledSurface *self = (MyPaintPythonTiledSurface *)tiled_surface;

    const gboolean readonly = request->readonly;
    const int tx = request->tx;
    const int ty = request->ty;

    if (PyErr_Occurred()) {
      PyErr_Print();
      return;
    }
    PyObject* rgba = PyObject_CallMethod(self->py_obj, "_get_tile_numpy", "(iii)", tx, ty, readonly);
    if (rgba == NULL) {
      request->buffer = NULL;
      printf("Python exception during get_tile_numpy()!\n");
      if (PyErr_Occurred())
        PyErr_Print();
      return;
    }

#ifdef HEAVY_DEBUG
    assert(PyArray_NDIM(rgba) == 3);
    assert(PyArray_DIM(rgba, 0) == tiled_surface->tile_size);
    assert(PyArray_DIM(rgba, 1) == tiled_surface->tile_size);
    assert(PyArray_DIM(rgba, 2) == 4);
    assert(PyArray_ISCARRAY(rgba));
    assert(PyArray_TYPE(rgba) == NPY_UINT16);
#endif
    // tiledsurface.py will keep a reference in its tiledict, at least until the final end_atomic()
    Py_DECREF(rgba);
    uint16_t * rgba_p = (uint16_t*)((PyArrayObject*)rgba)->data;

    request->buffer = rgba_p;
}

static void
tile_request_end(MyPaintTiledSurface *tiled_surface, MyPaintTiledSurfaceTileRequestData *request)
{
    // We modify tiles directly, so don't need to do anything here
}

MyPaintPythonTiledSurface *
mypaint_python_tiled_surface_new(PyObject *py_object)
{
    MyPaintPythonTiledSurface *self = (MyPaintPythonTiledSurface *)malloc(sizeof(MyPaintPythonTiledSurface));

    mypaint_tiled_surface_init(&self->parent, tile_request_start, tile_request_end);

    // MyPaintSurface vfuncs
    self->parent.parent.destroy = free_tiledsurf;
    self->parent.parent.begin_atomic = begin_atomic;
    self->parent.parent.end_atomic = end_atomic;

    self->py_obj = py_object; // no need to incref
    self->atomic = 0;

    return self;
}

void free_tiledsurf(MyPaintSurface *surface)
{
    MyPaintPythonTiledSurface *self = (MyPaintPythonTiledSurface *)surface;
    mypaint_tiled_surface_destroy(&self->parent);
    free(self);
}
