/* This file is part of MyPaint.
 * Copyright (C) 2008 by Martin Renold <martinxyz@gmx.ch>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License.
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY. See the COPYING file for more details.
 */

#include "Python.h"
#include <math.h>

#include "helpers.hpp"

#define TILE_SIZE 64

class TiledSurface {
  // the Python half of this calss is in tiledsurface.py
private:
  PyObject * self;
  Rect dirty_bbox;
  int atomic;
public:
  TiledSurface(PyObject * self_) {
    self = self_; // no need to incref
    atomic = 0;
    dirty_bbox.w = 0;
  }

  void begin_atomic() {
    if (atomic == 0) assert(dirty_bbox.w == 0);
    atomic++;
  }
  void end_atomic() {
    atomic--;
    if (atomic == 0) {
      Rect bbox = dirty_bbox; // copy to safety before calling python
      dirty_bbox.w = 0;
      if (bbox.w > 0) {
        PyObject* res;
        // OPTIMIZE: send a list tiles for minimal compositing? (but profile the code first)
        res = PyObject_CallMethod(self, "notify_observers", "(iiii)", bbox.x, bbox.y, bbox.w, bbox.h);
        if (!res) throw 0;
        Py_DECREF(res);
      }
    }
  }

  int draw_dab (float x, float y, 
                float radius, 
                float color_r, float color_g, float color_b,
                float opaque, float hardness = 0.5,
                float alpha_eraser = 1.0
                ) {

    float r_fringe;
    int xp, yp;
    float xx, yy, rr;
    float one_over_radius2;

    assert(color_r >= 0.0 && color_r <= 1.0);
    assert(color_g >= 0.0 && color_g <= 1.0);
    assert(color_b >= 0.0 && color_b <= 1.0);

    if (opaque == 0) return 0;
    if (radius < 0.1) return 0;
    if (hardness == 0) return 0; // infintly small point, rest transparent

    begin_atomic();

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
        PyObject* rgba;
        rgba = PyObject_CallMethod(self, "get_tile_memory", "(iii)", tx, ty, 0);
        if (!rgba) throw 0;

        assert(PyArray_NDIM(rgba) == 3);
        assert(PyArray_DIM(rgba, 0) == TILE_SIZE);
        assert(PyArray_DIM(rgba, 1) == TILE_SIZE);
        assert(PyArray_DIM(rgba, 2) == 4);

        assert(PyArray_ISCARRAY(rgba));
        assert(PyArray_TYPE(rgba) == NPY_UINT16);

        uint16_t * rgba_p  = (uint16_t*)((PyArrayObject*)rgba)->data;

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
              // In the formula below, topColor is assumed to be premultiplied.
              //
              //              opa_eraser  <   opa_       >
              // resultAlpha = topAlpha + (1.0 - topAlpha) * bottomAlpha
              // resultColor = topColor + (1.0 - topAlpha) * bottomColor
              //
              // (at least for the normal case where alpha_eraser == 1.0)
              // OPTIMIZE: separate function for the standard case without erasing

              //assert(opa >= 0.0 && opa <= 1.0);
              //assert(alpha_eraser >= 0.0 && alpha_eraser <= 1.0);

              uint32_t opa_a = (1<<15)*opa;   // topAlpha
              uint32_t opa_b = (1<<15)-opa_a; // bottomAlpha
              
              //uint16_t opa_eraser = 255 * opa * alpha_eraser;
              //assert(opa_ + opa_eraser <= 255);
              int idx = (yp*TILE_SIZE + xp)*4;

              // OPTIMIZE: don't use floats here in the inner loop?
              rgba_p[idx+3] = opa_a + (opa_b*rgba_p[idx+3])/(1<<15);
              rgba_p[idx+0] = (uint16_t)(color_r*opa_a) + opa_b*rgba_p[idx+0]/(1<<15);
              rgba_p[idx+1] = (uint16_t)(color_g*opa_a) + opa_b*rgba_p[idx+1]/(1<<15);
              rgba_p[idx+2] = (uint16_t)(color_b*opa_a) + opa_b*rgba_p[idx+2]/(1<<15);
            }
          }
        }
        Py_DECREF(rgba);
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

      ExpandRectToIncludePoint (&dirty_bbox, bb_x, bb_y);
      ExpandRectToIncludePoint (&dirty_bbox, bb_x+bb_w-1, bb_y+bb_h-1);
    }

    end_atomic();

    return 1;
  }

  void get_color (float x, float y, 
                  float radius, 
                  float * color_r, float * color_g, float * color_b, float * color_a
                  ) {
    *color_r = 0;
    *color_g = 0;
    *color_b = 0;
    *color_a = 0;
    // TODO
    /*
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
        tuple = PyObject_CallMethod(self, "get_tile_memory", "(iii)", tx, ty, 1);
        if (!tuple) throw 0;
        PyObject* rgb   = PyTuple_GET_ITEM(tuple, 0);
        PyObject* alpha = PyTuple_GET_ITEM(tuple, 1);
        Py_INCREF(rgb);
        Py_INCREF(alpha);
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
        Py_DECREF(rgb);
        Py_DECREF(alpha);
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
    */
  }
};

void composite_tile_over_rgb8(PyObject * src, PyObject * dst) {
  assert(PyArray_DIM(src, 0) == TILE_SIZE);
  assert(PyArray_DIM(src, 1) == TILE_SIZE);
  assert(PyArray_DIM(src, 2) == 4);
  assert(PyArray_TYPE(src) == NPY_UINT16);
  assert(PyArray_ISCARRAY(src));
  assert(PyArray_ISBEHAVED(src));

  assert(PyArray_DIM(dst, 0) == TILE_SIZE);
  assert(PyArray_DIM(dst, 1) == TILE_SIZE);
  assert(PyArray_DIM(dst, 2) == 3);
  assert(PyArray_TYPE(dst) == NPY_UINT8);
  assert(PyArray_ISCARRAY(dst));
  assert(PyArray_ISBEHAVED(src));
  
  uint16_t * src_p  = (uint16_t*)((PyArrayObject*)src)->data;
  uint8_t  * dst_p  = (uint8_t*) ((PyArrayObject*)dst)->data;
  int i;
  for (i=0; i<TILE_SIZE*TILE_SIZE; i++) {
    // resultAlpha = 1.0 (thus it does not matter if resultColor is premultiplied alpha or not)
    // resultColor = topColor + (1.0 - topAlpha) * bottomColor
    uint16_t * s = src_p + 4*i;
    uint8_t * d = dst_p + 3*i;
    const uint32_t one_minus_topAlpha = (1<<15) - s[3];
    d[0] = ((uint32_t)s[0]*255 + one_minus_topAlpha*d[0]) / (1<<15);
    d[1] = ((uint32_t)s[1]*255 + one_minus_topAlpha*d[1]) / (1<<15);
    d[2] = ((uint32_t)s[2]*255 + one_minus_topAlpha*d[2]) / (1<<15);
  }
}
