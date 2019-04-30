/* This file is part of MyPaint.
 * Copyright (C) 2018 by the MyPaint Development Team.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */

#ifndef FLOODFILL_HPP
#define FLOODFILL_HPP

#include <queue>
#include <vector>

#include "fill_common.hpp"

// Largest squared gap distance - represents infinite radius
#define MAX_GAP (2 * N * N)

// Enumeration of tile edges - used to determine seed range
// direction of origin, wrapped in a struct for SWIG's sake
struct edges {
    enum edge { north = 0, east = 1, south = 2, west = 3, none = 4 };
};

typedef edges::edge edge;

/*
  Implements the pixel threshold test function and uses it in the
  fill, alpha flooding, and tile uniformity/fillability methods
*/
class Filler
{
  private:
    const rgba targ;
    const rgba targ_premult;
    const fix15_t tolerance;
    std::queue<coord> queue;

  public:
    Filler(int targ_r, int targ_g, int targ_b, int targ_a, double tol);
    // Perform a scanline fill based on the rgba src tile and seed coordinates,
    // writing the resulting fill alphas to the dst tile and returning
    // any new overflows
    PyObject* fill(
        PyObject* src, PyObject* dst, PyObject* seeds, edge direction,
        int min_x, int min_y, int max_x, int max_y);
    // Flood the dst alpha tile with the fill alpha values
    // for the entire rgba src tile
    void flood(PyObject* src, PyObject* dst);
    // Test if the given rgba src tile is uniform
    // (all pixels having the same rgba color)), and if it is
    // returning the fill alpha for that color, otherwise Py_None
    PyObject* tile_uniformity(bool is_empty, PyObject* src);

  protected:
    // Pixel threshold test - the return value indicates the alpha value of the
    // filled pixel, where an alpha of 0 indicates that the pixel should not be
    // filled (and the fill not propagated through that pixel).
    chan_t pixel_fill_alpha(const rgba& src_px);
    bool check_enqueue(
        const int x, const int y, bool check, const rgba& src_px,
        const chan_t& dst_px);
};

// A GapClosingFiller uses additional distance data
// to stop filling when leaving a detected gap
class GapClosingFiller
{
  public:
    GapClosingFiller(int max_dist, bool track_seep);
    PyObject* fill(
        PyObject* alphas, PyObject* distances, PyObject* dst, PyObject* seeds,
        int min_x, int min_y, int max_x, int max_y);
    PyObject*
    unseep(PyObject* distances, PyObject* dst, PyObject* seeds, bool initial);

  protected:
    const int max_distance;
    const bool track_seep;
};

#endif //__HAVE_FLOODFILL_HPP
