/* This file is part of MyPaint.
 * Copyright (C) 2007 by Martin Renold <martinxyz@gmx.ch>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License.
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY. See the COPYING file for more details.
 */

#include <math.h>
#include "Python.h"
#include "Numeric/arrayobject.h"
#include "brush_dab.h"

// This actually draws every pixel of the dab.
// Called from brush_prepare_and_draw_dab.
// The parameters are only in the header file to avoid duplication.
{
  float r_fringe;
  int x0, y0;
  int x1, y1;
  int xp, yp;
  guchar *rgb;
  float xx, yy, rr;
  float radius2, one_over_radius2;
  //float precalc1, precalc2;
  //int m1, m2; 
  guchar randoms[8];
  guchar random_pos;

  guchar c[3];
  if (!s) return 0;
  
  g_assert (hardness <= 1.0 && hardness >= 0.0);
  if (hardness == 0) return 0; // infintly small point, rest transparent

  r_fringe = radius + 1;
  x0 = floor (x - r_fringe);
  y0 = floor (y - r_fringe);
  x1 = ceil (x + r_fringe);
  y1 = ceil (y + r_fringe);
  if (x0 < 0) x0 = 0;
  if (y0 < 0) y0 = 0;
  if (x1 > s->w) x1 = s->w;
  if (y1 > s->h) y1 = s->h;
  rr = SQR(radius);
  if (radius < 0.1) return 0;
  c[0] = color_r;
  c[1] = color_g;
  c[2] = color_b;
  radius2 = SQR(radius);
  one_over_radius2 = 1.0/radius2;
  
  // precalculate randomness
  ((guint32*)randoms)[0] = g_rand_int (rng);
  ((guint32*)randoms)[1] = g_rand_int (rng);
  random_pos = 0;

  g_assert (opaque >= 0 && opaque <= 1);
  //if (opaque == 0) return 0;
  if (opaque < 1/256.0) return 0;

  for (yp = y0; yp < y1; yp++) {
    yy = (yp + 0.5 - y);
    yy *= yy;
    for (xp = x0; xp < x1; xp++) {
      xx = (xp + 0.5 - x);
      xx *= xx;
      rr = (yy + xx) * one_over_radius2;
      // rr is in range 0.0..1.0*sqrt(2)
      rgb = PixelXY(s, xp, yp);

      if (rr <= 1.0) {
        int opa;
        { // hardness
          float o;
          if (hardness == 1) {
            o = 1.0;
          } else if (rr < hardness) {
            o = rr + 1-(rr/hardness);
            // hardness == 0 is nonsense, excluded above
          } else {
            o = hardness/(hardness-1)*(rr-1);
            // hardness == 1 ?
          }
          opa = opaque * o * 256 + 0.5;
          // comment out assertion: time-critical section
          //g_assert (opa >= 0 && opa <= 256);
          // opa is in range 0..256
        }
        
        int rgbdiff[3];
        int diff_sum;
        diff_sum = 0;
        rgbdiff[0] = c[0] - rgb[0]; diff_sum += ABS(rgbdiff[0]);
        rgbdiff[1] = c[1] - rgb[1]; diff_sum += ABS(rgbdiff[1]);
        rgbdiff[2] = c[2] - rgb[2]; diff_sum += ABS(rgbdiff[2]);
        // rgbdiff[] is in range -255..+255
        // dif_sum is in range 0..3*255

        rgbdiff[0] *= opa;
        rgbdiff[1] *= opa;
        rgbdiff[2] *= opa;
        // rgbdiff has range -255*256..+255*256

        int i;
        for (i=0; i<3; i++) {
          int reminder;
          int negative;
          if (rgbdiff[i] < 0) {
            negative = 1;
            rgbdiff[i] = - rgbdiff[i];
          } else {
            negative = 0;
          }
          // FIXME: ... 256? I think it is 255! Check this code again!
          // (compare it to other blending implementations, like gdkpixbuf-render.c)
          reminder = rgbdiff[i] % 256;
          rgbdiff[i] /= 256;
          // use randomness to fake more precision
          // - ah, I just learned that this is called "dither". I hope I've done it right.
          // FIXME: after the correction above, go and verify if this really helps.
          random_pos = (random_pos + 1 + rgbdiff[i] % 1 /* hope that's slightly random */) % 8;
          if (reminder > randoms[random_pos] /* 0..255 */) {
            rgbdiff[i]++;
          }
          if (negative) rgbdiff[i] = - rgbdiff[i];
        }
        
        rgb[0] += rgbdiff[0];
        rgb[1] += rgbdiff[1];
        rgb[2] += rgbdiff[2];
      }
      rgb += 3;
    }
  }
  
  if (bbox) {
    // expand the bounding box to include the region we just drawed
    int bb_x, bb_y, bb_w, bb_h;
    bb_x = floor (x - (radius+1));
    bb_y = floor (y - (radius+1));
    /* FIXME: think about it exactly */
    bb_w = ceil (2*(radius+1));
    bb_h = ceil (2*(radius+1));

    ExpandRectToIncludePoint (bbox, bb_x, bb_y);
    ExpandRectToIncludePoint (bbox, bb_x+bb_w-1, bb_y+bb_h-1);
  }

  return 1;
}


void draw_brush_dab_on_tiled_surface (PyObject * s, 
                                      GRand * rng,
                                      float x, float y, 
                                      float radius, float opaque, float hardness,
                                      float color_r, float color_g, float color_b
                                      ) {
  float r_fringe;
  int x0, y0;
  int x1, y1;
  int xp, yp;
  float xx, yy, rr;
  float radius2, one_over_radius2;

  if (opaque == 0) return;
  if (radius < 0.1) return;
  if (hardness == 0) return; // infintly small point, rest transparent

  r_fringe = radius + 1;
  rr = SQR(radius);


  int tx1 = floor(x - r_fringe) / TILE_SIZE;
  int tx2 = floor(x + r_fringe) / TILE_SIZE;
  int ty1 = floor(y - r_fringe) / TILE_SIZE;
  int ty2 = floor(y + r_fringe) / TILE_SIZE;
  int tx, ty;
  for (ty = ty1; ty <= ty2; ty++) {
    for (tx = tx1; tx <= tx2; tx++) {
      // OPTIMIZE: cache tile buffer pointers, so we don't have to call python for each dab;
      //           this could be used to return a list of dirty tiles at the same time
      //           (But profile this code first!)
      PyObject* tuple;
      tuple = PyObject_CallMethod(tiled_surface, "getTileMemory", "(ii)", tx, ty);
      if (!tuple) return;
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
        yy = (yp + 0.5 - y);
        yy *= yy;
        for (xp = x0; xp <= x1; xp++) {
          xx = (xp + 0.5 - x);
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
            color_r * opa;

            // We are manipulating pixels with premultiplied alpha directly.
            // This is an "over" operation (opa = topAlpha).
            //
            // resultAlpha = topAlpha + (1.0 - topAlpha) * bottomAlpha
            // resultColor = topColor + (1.0 - topAlpha) * bottomColor

            float opa_ = 1.0 - opa;
            int idx = yp*TILE_SIZE + xp;
            alpha_p[idx] = opa + opa_*alpha_p[idx]; 
            idx *= 3;
            rgb_p[idx+0] = color_r*opa + opa_*rgb_p[idx+0]; 
            rgb_p[idx+1] = color_g*opa + opa_*rgb_p[idx+1]; 
            rgb_p[idx+2] = color_b*opa + opa_*rgb_p[idx+2]; 
          }
        }
      }
    }
  }
}

