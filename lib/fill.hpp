/* This file is part of MyPaint.
 * Copyright (C) 2013-2014 by Andrew Chadwick <a.t.chadwick@gmail.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */

#ifndef FILL_HPP
#define FILL_HPP

#include <Python.h>

// Flood-fills one tile starting at a sequence of seed positions, and returns
// where overflows happened as seed lists in the cardinal directions.
//
// Returns a tuple of four lists (N, E, S, W) of overflow coordinate pairs
// [(x1, y1), ...] denoting to which pixels of the next tile in the identified
// direction the fill has overflowed. These coordinates can be fed back in to
// tile_flood_fill() for the tile identified as seeds.

PyObject *
tile_flood_fill (PyObject *src,     // readonly HxWx4 array of uint16
                 PyObject *dst,     // output HxWx4 array of uint16
                 PyObject *seeds,   // List of 2-tuples
                 int targ_r, int targ_g, int targ_b, int targ_a, //premult
                 double fill_r, double fill_g, double fill_b,
                 int min_x, int min_y, int max_x, int max_y,
                 double tolerance);       // [0..1]


#endif //__HAVE_FILL_HPP

