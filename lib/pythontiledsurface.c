/* This file is part of MyPaint.
 * Copyright (C) 2008-2011 by Martin Renold <martinxyz@gmx.ch>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */


#define TILE_SIZE 64
#define MAX_MIPMAP_LEVEL 4

// caching tile memory location (optimization)
#define TILE_MEMORY_SIZE 8
typedef struct {
  int tx, ty;
  uint16_t * rgba_p;
} TileMemory;

typedef struct {
    MyPaintTiledSurface parent;

    PyObject * py_obj;
    TileMemory tileMemory[TILE_MEMORY_SIZE];
    int tileMemoryValid;
    int tileMemoryWrite;
    int atomic;
    Rect dirty_bbox;
} MyPaintPythonTiledSurface;

// Forward declare
void free_tiledsurf(MyPaintSurface *surface);

void begin_atomic(MyPaintTiledSurface *tiled_surface)
{
    MyPaintPythonTiledSurface *self = (MyPaintPythonTiledSurface *)tiled_surface;

    if (self->atomic == 0) {
      assert(self->dirty_bbox.w == 0);
      assert(self->tileMemoryValid == 0);
    }
    self->atomic++;
}

void end_atomic(MyPaintTiledSurface *tiled_surface)
{
    MyPaintPythonTiledSurface *self = (MyPaintPythonTiledSurface *)tiled_surface;

    assert(self->atomic > 0);
    self->atomic--;

    if (self->atomic == 0) {
      self->tileMemoryValid = 0;
      self->tileMemoryWrite = 0;
      Rect bbox = self->dirty_bbox; // copy to safety before calling python
      self->dirty_bbox.w = 0;
      if (bbox.w > 0) {
        PyObject* res;
        // OPTIMIZE: send a list tiles for minimal compositing? (but profile the code first)
        res = PyObject_CallMethod(self->py_obj, "notify_observers", "(iiii)", bbox.x, bbox.y, bbox.w, bbox.h);
        if (!res) return;
        Py_DECREF(res);
      }
    }
}

uint16_t * get_tile_memory(MyPaintTiledSurface *tiled_surface, int tx, int ty, gboolean readonly)
{
    MyPaintPythonTiledSurface *self = (MyPaintPythonTiledSurface *)tiled_surface;

    // We assume that the memory location does not change between begin_atomic() and end_atomic().
    for (int i=0; i<self->tileMemoryValid; i++) {
      if (self->tileMemory[i].tx == tx and self->tileMemory[i].ty == ty) {
        return self->tileMemory[i].rgba_p;
      }
    }
    if (PyErr_Occurred()) return NULL;
    PyObject* rgba = PyObject_CallMethod(self->py_obj, "get_tile_memory", "(iii)", tx, ty, readonly);
    if (rgba == NULL) {
      printf("Python exception during get_tile_memory()!\n");
      return NULL;
    }
#ifdef HEAVY_DEBUG
       assert(PyArray_NDIM(rgba) == 3);
       assert(PyArray_DIM(rgba, 0) == TILE_SIZE);
       assert(PyArray_DIM(rgba, 1) == TILE_SIZE);
       assert(PyArray_DIM(rgba, 2) == 4);
       assert(PyArray_ISCARRAY(rgba));
       assert(PyArray_TYPE(rgba) == NPY_UINT16);
#endif
    // tiledsurface.py will keep a reference in its tiledict, at least until the final end_atomic()
    Py_DECREF(rgba);
    uint16_t * rgba_p = (uint16_t*)((PyArrayObject*)rgba)->data;

    // Cache tiles to speed up small brush strokes with lots of dabs, like charcoal.
    // Not caching readonly requests; they are alternated with write requests anyway.
    if (!readonly) {
      if (self->tileMemoryValid < TILE_MEMORY_SIZE) {
        self->tileMemoryValid++;
      }
      // We always overwrite the oldest cache entry.
      // We are mainly optimizing for strokes with radius smaller than one tile.
      self->tileMemory[self->tileMemoryWrite].tx = tx;
      self->tileMemory[self->tileMemoryWrite].ty = ty;
      self->tileMemory[self->tileMemoryWrite].rgba_p = rgba_p;
      self->tileMemoryWrite = (self->tileMemoryWrite + 1) % TILE_MEMORY_SIZE;
    }
    return rgba_p;
}

void update_tile(MyPaintTiledSurface *tiled_surface, int tx, int ty, uint16_t * tile_buffer)
{
    // We modify tiles directly, so don't need to do anything here
}

void area_changed(MyPaintTiledSurface *tiled_surface, int bb_x, int bb_y, int bb_w, int bb_h)
{
    MyPaintPythonTiledSurface *self = (MyPaintPythonTiledSurface *)tiled_surface;

    ExpandRectToIncludePoint (&self->dirty_bbox, bb_x, bb_y);
    ExpandRectToIncludePoint (&self->dirty_bbox, bb_x+bb_w-1, bb_y+bb_h-1);
}


MyPaintPythonTiledSurface *
mypaint_python_tiled_surface_new(PyObject *py_object)
{
    MyPaintPythonTiledSurface *self = (MyPaintPythonTiledSurface *)malloc(sizeof(MyPaintPythonTiledSurface));

    mypaint_tiled_surface_init(&self->parent);

    self->parent.parent.destroy = free_tiledsurf;

    self->parent.get_tile = get_tile_memory;
    self->parent.update_tile = update_tile;
    self->parent.begin_atomic = begin_atomic;
    self->parent.end_atomic = end_atomic;
    self->parent.area_changed = area_changed;

    self->py_obj = py_object; // no need to incref
    self->atomic = 0;
    self->tileMemoryValid = 0;
    self->tileMemoryWrite = 0;

    self->dirty_bbox.w = 0;
    self->dirty_bbox.h = 0;
    self->dirty_bbox.x = 0;
    self->dirty_bbox.y = 0;

    return self;
}

void free_tiledsurf(MyPaintSurface *surface)
{
    MyPaintPythonTiledSurface *self = (MyPaintPythonTiledSurface *)surface;
    mypaint_tiled_surface_destroy(&self->parent);
    free(self);
}
