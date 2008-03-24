/* This file is part of MyPaint.
 * Copyright (C) 2008 by Martin Renold <martinxyz@gmx.ch>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License.
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY. See the COPYING file for more details.
 */

extern "C" {
#include "Python.h"
#include "Numeric/arrayobject.h"
#include <math.h>
#include "helpers.h"
}

#define TILE_SIZE 64

// A RenderContext is normally created when we want to process one or
// more motion events, and destroyed when all of them are rendered and
// we are ready to display the result. (TODO: find a better name?)
struct RenderContext {
  PyObject * tiled_surface;
  Rect bbox;
  // planned: 
  // - list modified tiles?
  // - direct pointers to most recently used tiles (to avoid python call each dab)
};


int tile_draw_dab (RenderContext * rc,
                   float x, float y, 
                   float radius, 
                   float color_r, float color_g, float color_b,
                   float opaque, float hardness = 0.5,
                   float alpha_eraser = 1.0
                   ) {
  float r_fringe;
  int xp, yp;
  float xx, yy, rr;
  float one_over_radius2;

  if (opaque == 0) return 0;
  if (radius < 0.1) return 0;
  if (hardness == 0) return 0; // infintly small point, rest transparent

  r_fringe = radius + 1;
  rr = radius*radius;
  one_over_radius2 = 1.0/rr;

  int tx1 = floor(floor(x - r_fringe) / TILE_SIZE);
  int tx2 = floor(floor(x + r_fringe) / TILE_SIZE);
  int ty1 = floor(floor(y - r_fringe) / TILE_SIZE);
  int ty2 = floor(floor(y + r_fringe) / TILE_SIZE);
  int tx, ty;
  for (ty = ty1; ty <= ty2; ty++) {
    for (tx = tx1; tx <= tx2; tx++) {
      // OPTIMIZE: cache tile buffer pointers, so we don't have to call python for each dab;
      //           this could be used to return a list of dirty tiles at the same time
      //           (But profile this code first!)
      PyObject* tuple;
      tuple = PyObject_CallMethod(rc->tiled_surface, "getTileMemory", "(ii)", tx, ty);
      if (!tuple) return 0;
      PyObject* rgb   = PyTuple_GET_ITEM(tuple, 0);
      PyObject* alpha = PyTuple_GET_ITEM(tuple, 1);
      Py_DECREF(tuple);

      assert(PyArray_DIMS(rgb) == 3);
      assert(PyArray_DIM(rgb, 0) == TILE_SIZE);
      assert(PyArray_DIM(rgb, 1) == TILE_SIZE);
      assert(PyArray_DIM(rgb, 2) == 3);

      assert(PyArray_DIMS(alpha) == 3);
      assert(PyArray_DIM(alpha, 0) == TILE_SIZE);
      assert(PyArray_DIM(alpha, 1) == TILE_SIZE);
      assert(PyArray_DIM(alpha, 2) == 1);

      assert(ISCARRAY(rgb));
      assert(ISBEHAVED(rgb));
      assert(ISCARRAY(alpha));
      assert(ISBEHAVED(alpha));

      float * rgb_p   = (float*)((PyArrayObject*)rgb)->data;
      float * alpha_p = (float*)((PyArrayObject*)alpha)->data;

      float xc = x - tx*TILE_SIZE;
      float yc = y - ty*TILE_SIZE;

      int x0 = floor (xc - r_fringe);
      int y0 = floor (yc - r_fringe);
      int x1 = ceil (xc + r_fringe);
      int y1 = ceil (yc + r_fringe);
      if (x0 < 0) x0 = 0;
      if (y0 < 0) y0 = 0;
      if (x1 > TILE_SIZE-1) x1 = TILE_SIZE-1;
      if (y1 > TILE_SIZE-1) y1 = TILE_SIZE-1;

      for (yp = y0; yp <= y1; yp++) {
        yy = (yp + 0.5 - yc);
        yy *= yy;
        for (xp = x0; xp <= x1; xp++) {
          xx = (xp + 0.5 - xc);
          xx *= xx;
          rr = (yy + xx) * one_over_radius2;
          // rr is in range 0.0..1.0*sqrt(2)

          if (rr <= 1.0) {
            float opa = opaque;
            if (hardness < 1.0) {
              if (rr < hardness) {
                opa *= rr + 1-(rr/hardness);
                // hardness == 0 is nonsense, excluded above
              } else {
                opa *= hardness/(hardness-1)*(rr-1);
              }
            }

            // We are manipulating pixels with premultiplied alpha directly.
            // This is an "over" operation (opa = topAlpha).
            //
            // resultAlpha = topAlpha + (1.0 - topAlpha) * bottomAlpha
            // resultColor = topColor + (1.0 - topAlpha) * bottomColor
            //
            // (at least for the normal case where alpha_eraser == 1.0)
            // OPTIMIZE: separate function for the standard case without erasing

            float opa_ = 1.0 - opa;
            float opa_eraser = opa * alpha_eraser;
            int idx = yp*TILE_SIZE + xp;
            alpha_p[idx] = opa_eraser + opa_*alpha_p[idx]; 
            idx *= 3;
            rgb_p[idx+0] = color_r*opa_eraser + opa_*rgb_p[idx+0]; 
            rgb_p[idx+1] = color_g*opa_eraser + opa_*rgb_p[idx+1]; 
            rgb_p[idx+2] = color_b*opa_eraser + opa_*rgb_p[idx+2]; 
          }
        }
      }
    }
  }

  {
    // expand the bounding box to include the region we just drawed
    int bb_x, bb_y, bb_w, bb_h;
    bb_x = floor (x - (radius+1));
    bb_y = floor (y - (radius+1));
    /* FIXME: think about it exactly */
    bb_w = ceil (2*(radius+1));
    bb_h = ceil (2*(radius+1));

    ExpandRectToIncludePoint (&rc->bbox, bb_x, bb_y);
    ExpandRectToIncludePoint (&rc->bbox, bb_x+bb_w-1, bb_y+bb_h-1);
  }

  return 1;
}

void tile_get_color (PyObject * tiled_surface, 
                     float x, float y, 
                     float radius, 
                     float * color_r, float * color_g, float * color_b, float * color_a
                     ) {
  float r_fringe;
  int xp, yp;
  float xx, yy, rr;
  float one_over_radius2;

  if (radius < 1.0) radius = 1.0; // make sure we get at least one pixel
  const float hardness = 0.5;

  r_fringe = radius + 1;
  rr = radius*radius;
  one_over_radius2 = 1.0/rr;

  float sum_r, sum_g, sum_b, sum_a, sum_weight;
  sum_r = sum_g = sum_b = sum_a = sum_weight = 0.0;

  int tx1 = floor(floor(x - r_fringe) / TILE_SIZE);
  int tx2 = floor(floor(x + r_fringe) / TILE_SIZE);
  int ty1 = floor(floor(y - r_fringe) / TILE_SIZE);
  int ty2 = floor(floor(y + r_fringe) / TILE_SIZE);
  int tx, ty;
  for (ty = ty1; ty <= ty2; ty++) {
    for (tx = tx1; tx <= tx2; tx++) {
      PyObject* tuple;
      tuple = PyObject_CallMethod(tiled_surface, "getTileMemory", "(ii)", tx, ty);
      // OPTIMIZE: actually, we only need to get the tile for reading;
      // we could just continue if it is all transparent (and not
      // allocate new tile memory with the above call)
      assert (tuple); 
      PyObject* rgb   = PyTuple_GET_ITEM(tuple, 0);
      PyObject* alpha = PyTuple_GET_ITEM(tuple, 1);
      Py_DECREF(tuple);

      float * rgb_p   = (float*)((PyArrayObject*)rgb)->data;
      float * alpha_p = (float*)((PyArrayObject*)alpha)->data;

      float xc = x - tx*TILE_SIZE;
      float yc = y - ty*TILE_SIZE;

      int x0 = floor (xc - r_fringe);
      int y0 = floor (yc - r_fringe);
      int x1 = ceil (xc + r_fringe);
      int y1 = ceil (yc + r_fringe);
      if (x0 < 0) x0 = 0;
      if (y0 < 0) y0 = 0;
      if (x1 > TILE_SIZE-1) x1 = TILE_SIZE-1;
      if (y1 > TILE_SIZE-1) y1 = TILE_SIZE-1;

      for (yp = y0; yp <= y1; yp++) {
        yy = (yp + 0.5 - yc);
        yy *= yy;
        for (xp = x0; xp <= x1; xp++) {
          xx = (xp + 0.5 - xc);
          xx *= xx;
          rr = (yy + xx) * one_over_radius2;
          // rr is in range 0.0..1.0*sqrt(2)

          if (rr <= 1.0) {
            float opa = 1.0;
            if (hardness < 1.0) {
              if (rr < hardness) {
                opa *= rr + 1-(rr/hardness);
                // hardness == 0 is nonsense, excluded above
              } else {
                opa *= hardness/(hardness-1)*(rr-1);
              }
            }

            // note that we are working on premultiplied alpha
            // we do not un-premultiply it yet, so colors are weighted with their alpha
            int idx = yp*TILE_SIZE + xp;
            sum_weight += opa;
            sum_a      += opa*alpha_p[idx]; 
            idx *= 3;
            sum_r      += opa*rgb_p[idx+0];
            sum_g      += opa*rgb_p[idx+1];
            sum_b      += opa*rgb_p[idx+2];
          }
        }
      }
    }
  }

  assert(sum_weight > 0.0);
  *color_a = sum_a / sum_weight;
  // now un-premultiply the alpha
  if (sum_a > 0.0) {
    *color_r = sum_r / sum_a;
    *color_g = sum_g / sum_a;
    *color_b = sum_b / sum_a;
  } else {
    // it is all transparent, so don't care about the colors
    // (let's make them ugly so bugs will be visible)
    *color_r = 0.0;
    *color_g = 1.0;
    *color_b = 0.0;
  }
  assert (*color_a >= 0.0);
  assert (*color_r >= 0.0);
  assert (*color_g >= 0.0);
  assert (*color_b >= 0.0);
  assert (*color_r <= 1.001);
  assert (*color_g <= 1.001);
  assert (*color_b <= 1.001);
}

/* unused code to render an array of dabs
PyObject * render (PyObject * layer, PyObject * dabs) {

  dabs = PyArray_ContiguousFromObject(dabs, PyArray_FLOAT, 2, 2);
  if (!dabs) return NULL;

  PyArrayObject * array = (PyArrayObject*)dabs;

  int n = array->dimensions[0];
  assert(array->dimensions[1] == 8);

  Dab * d = (Dab*)array->data;
  int i;
  for (i=0; i<n; i++) {

    draw_brush_dab_on_tiled_surface (layer,
                                     d[i].x, d[i].y, 
                                     d[i].radius, d[i].opaque, d[i].hardness,
                                     d[i].r, d[i].g, d[i].b
                                     );
  }

  Py_DECREF(dabs);

  Py_RETURN_NONE;
}
*/
