/* This file is part of MyPaint.
 * Copyright (C) 2008 by Martin Renold <martinxyz@gmx.ch>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */

const int ccdb_size = 256;

class ColorChangerCrossedBowl {
public:
  float brush_h, brush_s, brush_v;
  void set_brush_color(float h, float s, float v)
  {
    brush_h = h;
    brush_s = s;
    brush_v = v;
  }

  int get_size() 
  {
    return ccdb_size;
  }

#ifndef SWIG

  struct PrecalcData {
    int h;
    int s;
    int v;
    //signed char s;
    //signed char v;
  };

  PrecalcData * precalcData[4];
  int precalcDataIndex;

  ColorChangerCrossedBowl()
  {
    precalcDataIndex = -1;
    for (int i=0; i<4; i++) {
      precalcData[i] = NULL;
    }
  }

  PrecalcData * precalc_data(float phase0)
  {
    // Hint to the casual reader: some of the calculation here do not
    // what I originally intended. Not everything here will make sense.
    // It does not matter in the end, as long as the result looks good.

    int width, height;
    int x, y, i;
    int s_radius = ccdb_size/2.6;
    PrecalcData * result;

    width = ccdb_size;
    height = ccdb_size;
    result = (PrecalcData*)malloc(sizeof(PrecalcData)*width*height);

    i = 0;
    for (y=0; y<height; y++) {
      for (x=0; x<width; x++) {
        float v_factor = 0.6;
        float s_factor = 0.6;

#define factor2_func(x) ((x)*(x)*SIGN(x))
#define SQR2(x) (x)*(x)/2 + (x)/2
        float v_factor2 = 0.013;
        float s_factor2 = 0.013;

        int stripe_width = 15;

        float h = 0;
        float s = 0;
        float v = 0;

        int dx = x-width/2;
        int dy = y-height/2;
        int diag = sqrt(2)*ccdb_size/2;

        int dxs, dys;
        if (dx > 0) 
            dxs = dx - stripe_width;
        else
            dxs = dx + stripe_width;
        if (dy > 0) 
            dys = dy - stripe_width;
        else
            dys = dy + stripe_width;

        float r = sqrt(SQR(dxs)+SQR(dys));

        // hue
        if (r < s_radius) {
            if (dx > 0) 
                h = 90*SQR2(r/s_radius);
            else
                h = 360 - 90*SQR2(r/s_radius);
            s = 256*(atan2f(abs(dxs),dys)/M_PI) - 128;
        } else {
            h = 180 + 180*atan2f(dys,-dxs)/M_PI;
            v = 255*(r-s_radius)/(diag-s_radius) - 128;
        }

        // horizontal and vertical lines
        int min = ABS(dx);
        if (ABS(dy) < min) min = ABS(dy);
        if (min < stripe_width) {
          h = 0;
          // x-axis = value, y-axis = saturation
          v =    dx*v_factor + factor2_func(dx)*v_factor2;
          s = - (dy*s_factor + factor2_func(dy)*s_factor2);
          // but not both at once
          if (ABS(dx) > ABS(dy)) {
            // horizontal stripe
            s = 0.0;
          } else {
            // vertical stripe
            v = 0.0;
          }
        } else {
          // diagonal lines
          min = ABS(dx+dy);
          if (ABS(dx-dy) < min) min = ABS(dx-dy);
          if (min < stripe_width) {
            h = 0;
            // x-axis = value, y-axis = saturation
            v =    dx*v_factor + factor2_func(dx)*v_factor2;
            s = - (dy*s_factor + factor2_func(dy)*s_factor2);
            // both at once
          }
        }

        result[i].h = (int)h;
        result[i].v = (int)v;
        result[i].s = (int)s;
        i++;
      }
    }
    return result;
  }

  void get_hsv(float &h, float &s, float &v, PrecalcData * pre)
  {
    h = brush_h + pre->h/360.0;
    s = brush_s + pre->s/255.0;
    v = brush_v + pre->v/255.0;

    h -= floor(h);
    s = CLAMP(s, 0.0, 1.0);
    v = CLAMP(v, 0.0, 1.0);

  }

#endif /* #ifndef SWIG */

  void render(PyObject * obj)
  {
    uint8_t * pixels;
    int x, y;
    float h, s, v;

    PyArrayObject* arr = (PyArrayObject*)obj;

    assert(PyArray_ISCARRAY(arr));
    assert(PyArray_NDIM(arr) == 3);
    assert(PyArray_DIM(arr, 0) == ccdb_size);
    assert(PyArray_DIM(arr, 1) == ccdb_size);
    assert(PyArray_DIM(arr, 2) == 4);
    pixels = (uint8_t*)PyArray_DATA(arr);
    
    precalcDataIndex++;
    precalcDataIndex %= 4;

    PrecalcData * pre = precalcData[precalcDataIndex];
    if (!pre) {
      pre = precalcData[precalcDataIndex] = precalc_data(2*M_PI*(precalcDataIndex/4.0));
    }

    for (y=0; y<ccdb_size; y++) {
      for (x=0; x<ccdb_size; x++) {

        get_hsv(h, s, v, pre);
        pre++;

        hsv_to_rgb_range_one (&h, &s, &v);
        uint8_t * p = pixels + 4*(y*ccdb_size + x);
        p[0] = h; p[1] = s; p[2] = v; p[3] = 255;
      }
    }
  }

  PyObject* pick_color_at(float x_, float y_)
  {
    float h,s,v;
    PrecalcData * pre = precalcData[precalcDataIndex];
    assert(precalcDataIndex >= 0);
    assert(pre != NULL);
    int x = CLAMP(x_, 0, ccdb_size);
    int y = CLAMP(y_, 0, ccdb_size);
    pre += y*ccdb_size + x;
    get_hsv(h, s, v, pre);
    return Py_BuildValue("fff",h,s,v);
  }
};
