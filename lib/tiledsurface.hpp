/* This file is part of MyPaint.
 * Copyright (C) 2008-2011 by Martin Renold <martinxyz@gmx.ch>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */

#include <mypaint-tiled-surface.h>

// Used by pythontiledsurface.c,
// needs to be defined here so that swig and the Python bindings will find it
#define TILE_SIZE 64
#define MAX_MIPMAP_LEVEL 4

// Implementation of tiled surface backend
#include "pythontiledsurface.c"

// Interface class, wrapping the backend the way MyPaint wants to use it
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
      mypaint_surface_begin_atomic((MyPaintSurface *)c_surface);
  }
  void end_atomic() {
      mypaint_surface_end_atomic((MyPaintSurface *)c_surface);
  }

  uint16_t * get_tile_memory(int tx, int ty, bool readonly) {
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
