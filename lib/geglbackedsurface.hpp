/* This file is part of MyPaint.
 * Copyright (C) 2012 by Jon Nordby <jononor@gmail.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */

#include <gegl.h>
#include <gegl-utils.h>
#include <pygobject.h>

#include <vector>

#include <mypaint-gegl-surface.h>

class GeglBackedSurface : public Surface {

public:
  GeglBackedSurface(PyObject * self_) {
      py_obj = self_; // no need to incref

      // TODO: move somewhere else
      babl_init();
      gegl_init(0, NULL);
      pygobject_init(2, 18, 0);

      c_surface = mypaint_gegl_tiled_surface_new();

      node = gegl_node("gegl:buffer-source",
                       "buffer", mypaint_gegl_tiled_surface_get_buffer(c_surface), NULL);
      g_assert(node);
      py_node = pygobject_new(G_OBJECT(node));
  }

  ~GeglBackedSurface() {
      mypaint_surface_destroy((MyPaintSurface *)c_surface);

      g_object_unref(node);
      Py_DECREF(py_node);
  }

  void begin_atomic() {
      mypaint_tiled_surface_begin_atomic((MyPaintTiledSurface *)c_surface);
  }
  void end_atomic() {
      mypaint_tiled_surface_end_atomic((MyPaintTiledSurface *)c_surface);
  }

  uint16_t * get_tile_memory(int tx, int ty, bool readonly) {
      return mypaint_tiled_surface_get_tile((MyPaintTiledSurface *)c_surface, tx, ty, readonly);
  }

  // returns true if the surface was modified
  virtual bool draw_dab (float x, float y,
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

  virtual void get_color (float x, float y,
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

  std::vector<int> get_bbox_c() {

      // Only reason to use vector is so that swig can automatically map it to a tuple
      std::vector<int> bbox;

      GeglRectangle extent_rect;
      extent_rect = *gegl_buffer_get_extent(mypaint_gegl_tiled_surface_get_buffer(c_surface));

      bbox.push_back(extent_rect.x);
      bbox.push_back(extent_rect.y);
      bbox.push_back(extent_rect.width);
      bbox.push_back(extent_rect.height);

      return bbox;
  }

  // Return a PyGObject for the GeglNode
  PyObject* get_node() {
      return py_node;
  }

  void save_as_png_c(const char *path) {

      GeglNode *graph, *save, *source;

      graph = gegl_node_new();
      source = gegl_node_new_child(graph, "operation", "gegl:buffer-source",
                                   "buffer", mypaint_gegl_tiled_surface_get_buffer(c_surface), NULL);
      save = gegl_node_new_child(graph, "operation", "gegl:png-save", "path", path, NULL);
      gegl_node_link(source, save);

      gegl_node_process(save);
  }

  void load_from_png_c(const char *path) {
      GeglNode *graph, *load, *sink;

      graph = gegl_node_new();
      load = gegl_node_new_child(graph, "operation", "gegl:png-load", "path", path, NULL);
      sink = gegl_node_new_child(graph, "operation", "gegl:buffer-sink",
                                 "buffer", mypaint_gegl_tiled_surface_get_buffer(c_surface), NULL);
      gegl_node_link(load, sink);

      gegl_node_process(sink);
  }

private:
    MyPaintGeglTiledSurface *c_surface;
    PyObject * py_obj;
    GeglNode *node;
    PyObject *py_node;
};
