/* This file is part of MyPaint.
 * Copyright (C) 2008 by Martin Renold <martinxyz@gmx.ch>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License.
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY. See the COPYING file for more details.
 */

// surface interface required by brush.hpp
class Surface {
public:
  virtual bool draw_dab (float x, float y, 
                         float radius, 
                         float color_r, float color_g, float color_b,
                         float opaque, float hardness = 0.5,
                         float alpha_eraser = 1.0
                         ) = 0;

  virtual void get_color (float x, float y, 
                          float radius, 
                          float * color_r, float * color_g, float * color_b, float * color_a
                          ) = 0;
};
