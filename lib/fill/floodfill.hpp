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

#include "fill_common.hpp"

/*
  Enumeration of tile edges - used to determine seed range
  direction of origin.
  It is wrapped in a struct for SWIG's sake
*/
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
    const rgba target_color;
    const rgba target_color_premultiplied;
    const fix15_t tolerance;
    std::queue<coord> seed_queue;

  public:
    Filler(int targ_r, int targ_g, int targ_b, int targ_a, double tol);
    // Perform a scanline fill based on the rgba src tile and seed coordinates,
    // writing the resulting fill alphas to the dst tile and returning
    // any new overflows
    PyObject* fill(
        PyObject* src, PyObject* dst, PyObject* seeds, edge direction,
        int min_x, int min_y, int max_x, int max_y);
    // Flood the dst alpha tile with the fill alpha values
    // for the entire rgba src tile, relative to the target color
    void flood(PyObject* src, PyObject* dst);
    // Test if the given rgba src tile is uniform
    // (all pixels having the same rgba color)), and if it is
    // returning the fill alpha for that color, otherwise Py_None
    PyObject* tile_uniformity(bool is_empty, PyObject* src);

  private:
    // Pixel threshold test - the return value indicates the alpha value of the
    // filled pixel, where an alpha of 0 indicates that the pixel should not be
    // filled (and the fill not propagated through that pixel).
    chan_t pixel_fill_alpha(const rgba& src_px);
    // Queue seeds from a python list of (x, y) coordinate tuples
    void queue_seeds(
        PyObject* seeds, PixelBuffer<rgba>& src, PixelBuffer<chan_t> dst);
    // Queue seeds from a python list of [start, end] range tuples
    // paired with an input origin direction indicating the side of
    // the tile that the ranges apply to.
    // Ranges are left->right, top->down, and end-inclusive.
    void queue_ranges(
        edge direction, PyObject* seeds, bool marks[N],
        PixelBuffer<rgba>& src, PixelBuffer<chan_t>& dst);
    // Check if a pixel is a valid fill candidate (unfilled & within threshold)
    // Put it in the seed queue if true.
    // Return value means: "enqueue valid neighbours on same row".
    bool check_enqueue(
        const int x, const int y, bool check, const rgba& src_px,
        const chan_t& dst_px);
};

/*
  A GapClosingFiller uses additional distance data
  to stop filling when leaving a detected gap
*/
class GapClosingFiller
{
  public:
    GapClosingFiller(int max_dist, bool track_seep);
    // Perform a gap closing fill based on an input alpha tile, a tile with
    // pixels marked with detected distances and input seeds with distance
    // data points.
    PyObject* fill(
        PyObject* alphas, PyObject* distances, PyObject* dst, PyObject* seeds,
        int min_x, int min_y, int max_x, int max_y);
    // Progressively erase a fill until reaching a point where the detected
    // is a certain amount larger than the smallest detected distance
    PyObject*
    unseep(PyObject* distances, PyObject* dst, PyObject* seeds, bool initial);

  protected:
    const int max_distance;
    const bool track_seep;
};

/*
  Create and return an N x N rgba tile based on an rgb color
  and a N x N tile of alpha values
*/
PyObject* rgba_tile_from_alpha_tile(
    PyObject* src, double fill_r, double fill_g, double fill_b, int min_x,
    int min_y, int max_x, int max_y);

#endif //FLOODFILL_HPP
