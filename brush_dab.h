/* This file is part of MyPaint.
 * Copyright (C) 2007 by Martin Renold <martinxyz@gmx.ch>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License.
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY. See the COPYING file for more details.
 */

#include <gtk/gtk.h>
#include "gtkmysurfaceold.h"
#include "helpers.h"

// The bbox (bounding box) can be NULL, if not, it will be expanded to
// include the surface area which was just painted.
// Returns 0 if nothing was painted.
int draw_brush_dab (GtkMySurfaceOld * s, Rect * bbox,
                    GRand * rng,
                    float x, float y, 
                    float radius, float opaque, float hardness,
                    int color_r, int color_g, int color_b
                    )
     // ; follows after #include
