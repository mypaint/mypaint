/* This file is part of MyPaint.
 * Copyright (C) 2008-2011 by Martin Renold <martinxyz@gmx.ch>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */

#include <mypaint-tiled-surface.h>

enum SymmetryType
{
        SymmetryVertical,
        SymmetryHorizontal,
        SymmetryVertHorz,
        SymmetryRotational,
        SymmetrySnowflake,
        NumSymmetryTypes
};

static const int TILE_SIZE = MYPAINT_TILE_SIZE;
static const int MAX_MIPMAP_LEVEL = MYPAINT_MAX_MIPMAP_LEVEL;

// Implementation of tiled surface backend
#include "pythontiledsurface.cpp"

#include <vector>

// Interface class, wrapping the backend the way MyPaint wants to use it
class TiledSurface : public Surface {
  // the Python half of this class is in tiledsurface.py

public:
  TiledSurface(PyObject * self_) {
      c_surface = mypaint_python_tiled_surface_new(self_);
      tile_request_in_progress = false;
  }

  ~TiledSurface() {
      mypaint_surface_unref((MyPaintSurface *)c_surface);
  }

  void set_symmetry_state(bool active,
        float center_x, float center_y,
        enum SymmetryType symmetry_type, int rot_symmetry_lines) {
    mypaint_tiled_surface_set_symmetry_state((MyPaintTiledSurface *)c_surface, active,
        center_x, center_y,
        (MyPaintSymmetryType)symmetry_type, rot_symmetry_lines);
  }

  void begin_atomic() {
      mypaint_surface_begin_atomic((MyPaintSurface *)c_surface);
  }
  std::vector<int> end_atomic() {
      MyPaintRectangle bbox_rect;
      mypaint_surface_end_atomic((MyPaintSurface *)c_surface, &bbox_rect);
      std::vector<int> bbox = std::vector<int>(4, 0);
      bbox[0] = bbox_rect.x;     bbox[1] = bbox_rect.y;
      bbox[2] = bbox_rect.width; bbox[3] = bbox_rect.height;
      return bbox;
  }

  // returns true if the surface was modified
  // Note: Used only in test_mypaintlib.py
  bool draw_dab (float x, float y, 
                 float radius, 
                 float color_r, float color_g, float color_b,
                 float opaque, float hardness = 0.5,
                 float color_a = 1.0,
                 float aspect_ratio = 1.0, float angle = 0.0,
                 float lock_alpha = 0.0,
                 float colorize = 0.0,
                 float posterize = 0.0,
                 float posterize_num = 0.0,
                 float paint = 1.0
                 ) {

    return mypaint_surface_draw_dab((MyPaintSurface *)c_surface, x, y, radius, color_r, color_g, color_b,
                             opaque, hardness, color_a, aspect_ratio, angle,
                             lock_alpha, colorize, posterize, posterize_num, paint);
  }

  std::vector<double> get_color (double x, double y, double radius) {
    std::vector<double> rgba = std::vector<double>(4, 0.0);
    float r,g,b,a,paint;
    paint = 1.0;
    mypaint_surface_get_color((MyPaintSurface *)c_surface, x, y, radius,
                              &r, &g, &b, &a, paint);
    rgba[0] = r; rgba[1] = g; rgba[2] = b; rgba[3] = a;
    return rgba;
  }

  float get_alpha (float x, float y, float radius) {
      return mypaint_surface_get_alpha((MyPaintSurface *)c_surface, x, y, radius);
  }

  MyPaintSurface *get_surface_interface() {
    return (MyPaintSurface*)c_surface;
  }

private:
    MyPaintPythonTiledSurface *c_surface;
    MyPaintTileRequest tile_request;
    bool tile_request_in_progress;
};

static PyObject *
get_module(char *name)
{
    PyObject *pName = PyString_FromString(name);
    PyObject *pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    if (pModule != NULL) {

    }
    else {
        PyErr_Print();
        fprintf(stderr, "Failed to load \"%s\"\n", name);
        return NULL;
    }
    return pModule;
}

static PyObject *
new_py_tiled_surface(PyObject *pModule)
{
    PyObject *pFunc = PyObject_GetAttrString(pModule, "_new_backend_surface");

    assert(pFunc && PyCallable_Check(pFunc));

    PyObject *pArgs = PyTuple_New(0);
    PyObject *pValue = PyObject_CallObject(pFunc, pArgs);
    Py_DECREF(pArgs);

    return pValue;
}

extern "C" {

MyPaintSurface *
mypaint_python_surface_factory(gpointer user_data)
{
    PyObject *module = get_module("lib.tiledsurface");
    PyObject *instance = new_py_tiled_surface(module);
    assert(instance != NULL);
    // Py_DECREF(module);

    static const char *type_str = "TiledSurface *";
    swig_type_info *info = SWIG_TypeQuery(type_str);
    if (! info) {
        fprintf(stderr, "SWIG_TypeQuery failed to look up '%s'", type_str);
        return NULL;
    }
    TiledSurface *surf;
    if (SWIG_ConvertPtr(instance, (void **)&surf, info, SWIG_POINTER_EXCEPTION) == -1) {
        fprintf(stderr, "SWIG_ConvertPtr failed\n");
        return NULL;
    }
    MyPaintSurface *interface = surf->get_surface_interface();

    // Py_DECREF(instance);

    return interface;
}

} // extern "C"
