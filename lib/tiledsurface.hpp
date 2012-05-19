/* This file is part of MyPaint.
 * Copyright (C) 2008-2011 by Martin Renold <martinxyz@gmx.ch>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */

#include <mypaint-tiled-surface.h>

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

uint16_t * get_tile_memory(MyPaintTiledSurface *tiled_surface, int tx, int ty, bool readonly)
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

class TiledSurface : public Surface {
  // the Python half of this class is in tiledsurface.py

public:
  TiledSurface(PyObject * self_) {
      c_surface = mypaint_python_tiled_surface_new(self_);
  }

  ~TiledSurface() {
      mypaint_surface_destroy((MyPaintSurface *)c_surface);
  }

  void set_symmetry_state(bool active, float center_x) {
    mypaint_tiled_surface_set_symmetry_state((MyPaintTiledSurface *)c_surface, active, center_x);
  }

  void begin_atomic() {
      mypaint_tiled_surface_begin_atomic((MyPaintTiledSurface *)c_surface);
  }
  void end_atomic() {
      mypaint_tiled_surface_end_atomic((MyPaintTiledSurface *)c_surface);
  }

  virtual uint16_t * get_tile_memory(int tx, int ty, bool readonly) {
      return mypaint_tiled_surface_get_tile((MyPaintTiledSurface *)c_surface, tx, ty, readonly);
  }

  // returns true if the surface was modified
  bool draw_dab (float x, float y, 
                 float radius, 
                 float color_r, float color_g, float color_b,
                 float opaque, float hardness = 0.5,
                 float color_a = 1.0,
                 float aspect_ratio = 1.0, float angle = 0.0,
                 float lock_alpha = 0.0,
                 float colorize = 0.0,
                 int recursing = 0 // used for symmetry, internal use only
                 ) {

    return mypaint_surface_draw_dab((MyPaintSurface *)c_surface, x, y, radius, color_r, color_g, color_b,
                             opaque, hardness, color_a, aspect_ratio, angle,
                             lock_alpha, colorize);
  }

  void get_color (float x, float y, 
                  float radius, 
                  float * color_r, float * color_g, float * color_b, float * color_a
                  ) {
    mypaint_surface_get_color((MyPaintSurface *)c_surface, x, y, radius, color_r, color_g, color_b, color_a);
  }

  float get_alpha (float x, float y, float radius) {
      return mypaint_surface_get_alpha((MyPaintSurface *)c_surface, x, y, radius);
  }

  MyPaintSurface *get_surface_interface() {
    return (MyPaintSurface*)c_surface;
  }

private:
    MyPaintPythonTiledSurface *c_surface;
};
