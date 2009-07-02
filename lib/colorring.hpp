/* This file is part of MyPaint.
 * Copyright (C) 2008 by Clement Skau
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */

#include <cmath> // atan2, sqrt or hypot
#include "helpers2.hpp"

class SCWSColorSelector {
public:
  static const int size = 240; // diameter of Swiss Cheese Wheel Color Selector(TM)
  static const int center = (size/2); // radii/center coordinate of SCWCS

  /*
    --------- Swiss Cheese Wheel Color Selector(TM) --------- 
  
    Ring 0: Current brush color
    Ring 1: Value
    Ring 2: Saturation
    Ring 3: Hue
    
  */
  
  // Frequently used constants
  static const float RAD_TO_ONE = 0.5f/M_PI;
  static const float TWO_PI = 2.0f*M_PI;
  // Calculate these as precise as the hosting system can once and for all
  static const float ONE_OVER_THREE = 1.0f/3.0f;
  static const float TWO_OVER_THREE = 2.0f/3.0f;
  
  float brush_h, brush_s, brush_v;
  void set_brush_color(float h, float s, float v)
  {
    brush_h = h;
    brush_s = s;
    brush_v = v;
  }

  // 1 Mile of variables....
  void get_hsva_at( float* h, float* s, float* v, float* a, float x, float y, bool adjust_color = true, bool only_colors = true, float mark_h = 0.0f )
  {
    float rel_x = (center-x);
    float rel_y = (center-y);
    
    //float radi = sqrt( rel_x*rel_x + rel_y*rel_y ); // Pre-C99 solution
    float radi = hypot( rel_x, rel_y );
    float theta = atan2( rel_y, rel_x );
    if( theta < 0.0f ) theta += TWO_PI; // Range: [ 0, 2*PI )
  
    // Current brush color
    *h = brush_h;
    *s = brush_s;
    *v = brush_v;
    *a = 255.0f; // Alpha is always [0,255]
  
    if( radi > 120.0f ) // Masked/Clipped/Tranparent area
      {
        // transparent/cut away
        *a = 0.0f;
      }
    else if( radi <= 27.0f ) // center disk
      {
        // exit by clicking
        if (only_colors) *a = 0.0f;
      }
    else if( radi > 27.0f && radi <= 30.0f ) // white border
      {
        *h = *s = 0.0f;
        *v = 1.0f;
      }
    else if( radi > 30.0f && radi <= 55.0f ) // Saturation
      {
        *s = (theta/TWO_PI);
    
        if( only_colors == false && floor(*s*255.0f) == floor(brush_s*255.0f) ) {
          // Draw marker
          *s = *v = 1.0f;
          *h = mark_h;
        }
    
      }
    else if( radi > 55.0f && radi <= 85.0f ) // Value 
      {
        *v = (theta/TWO_PI);
    
        if( only_colors == false && floor(*v*255.0f) == floor(brush_v*255.0f) ) {
          // Draw marker
          *s = *v = 1.0f;
          *h = mark_h;
        }
    
      }
    else if( radi > 85.0f && radi <= 120.0f ) // Hue
      {
        *h = (theta*RAD_TO_ONE);
    
        if( only_colors == false && floor(*h*360.0f) == floor(brush_h*360.0f) ) {
          // Draw marker
          *h = mark_h;
        }
    
        if( adjust_color == false ) {
          // Picking a new hue resets Saturation and Value
          *s = *v = 1.0f;
        }
      }
  }

  PyObject* pick_color_at( float x, float y)
  {
    float h,s,v,a;
    get_hsva_at(&h, &s, &v, &a, x, y);
    if (a == 0) {
      Py_RETURN_NONE;
    }
    return Py_BuildValue("fff",h,s,v);
  }

  void render(PyObject * arr)
  {
    assert(PyArray_ISCARRAY(arr));
    assert(PyArray_NDIM(arr) == 3);
    assert(PyArray_DIM(arr, 0) == size);
    assert(PyArray_DIM(arr, 1) == size);
    assert(PyArray_DIM(arr, 2) == 4);  // memory width of pixel data ( 3 = RGB, 4 = RGBA )
    guchar* pixels = (guchar*)((PyArrayObject*)arr)->data;
  
    const int pixels_inc = PyArray_DIM(arr, 2);
  
    float h,s,v,a;
  
    float ofs_h = ((brush_h+ONE_OVER_THREE)>1.0f)?(brush_h-TWO_OVER_THREE):(brush_h+ONE_OVER_THREE); // offset hue

    for(float y=0; y<size; y++) {
      for(float x=0; x<size; x++) {
        get_hsva_at(&h, &s, &v, &a, x, y, false, false, ofs_h);
        hsv_to_rgb_range_one(&h,&s,&v); // convert from HSV [0,1] to RGB [0,255]
        pixels[0] = h; pixels[1] = s; pixels[2] = v; pixels[3] = a;
        pixels += pixels_inc; // next pixel block
      }
    }
  }
};
