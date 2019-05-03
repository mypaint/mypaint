/* This file is part of MyPaint.
 * Copyright (C) 2019 by the MyPaint Development Team.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */

#include "fill_common.hpp"

PyObject* ConstTiles::_ALPHA_TRANSPARENT = nullptr;
PyObject* ConstTiles::_ALPHA_OPAQUE = nullptr;

void
ConstTiles::init()
{
    npy_intp dims[] = {N, N};
    PyObject* empty = PyArray_ZEROS(2, dims, NPY_USHORT, false);
    PyObject* full = PyArray_EMPTY(2, dims, NPY_USHORT, false);
    PixelBuffer<chan_t> buf{full};
    PixelRef<chan_t> ref = buf.get_pixel(0, 0);
    for (int i = 0; i < N * N; ++i, ref.move_x(1)) {
        ref.write(fix15_one);
    }
    _ALPHA_TRANSPARENT = empty;
    _ALPHA_OPAQUE = full;
}

PyObject*
ConstTiles::ALPHA_TRANSPARENT()
{
    if (!_ALPHA_TRANSPARENT) init();
    return _ALPHA_TRANSPARENT;
}

PyObject*
ConstTiles::ALPHA_OPAQUE()
{
    if (!_ALPHA_OPAQUE) init();
    return _ALPHA_OPAQUE;
}

/*
  Helper function to copy a rectangular slice of the input
  buffer to the full input array.
*/
static void
init_rect(
    const int x, const int w, const int y, const int h,
    PixelBuffer<chan_t> input_buf, chan_t** input, const int px_x,
    const int px_y)
{
    PixelRef<chan_t> in_px = input_buf.get_pixel(px_x, px_y);
    for (int y_i = y; y_i < y + h; ++y_i) {
        for (int x_i = x; x_i < x + w; ++x_i) {
            input[y_i][x_i] = in_px.read();
            in_px.move_x(1);
        }
        in_px.move_x(0 - w);
        in_px.move_y(1);
    }
}

void
init_from_nine_grid(
    int radius, chan_t** input, bool from_above, GridVector grid)
{
    const int r = radius;

// Using macro here to avoid performance hit on gcc <= 5.4
#define B (N - r)
#define E (N + r)
    if (from_above) {
        // Reuse radius*2 rows from previous morph
        // and no need to handle the topmost tiles
        for (int i = 0; i < r * 2; ++i) {
            chan_t* tmp = input[i];
            input[i] = input[N + i];
            input[N + i] = tmp;
        } // west, mid, east: bottom (N-r) rows
        init_rect(0, r, 2 * r, B, grid[3], input, B, r);
        init_rect(r, N, 2 * r, B, grid[4], input, 0, r);
        init_rect(E, r, 2 * r, B, grid[5], input, 0, r);
    } else { // nw, north, ne
        init_rect(0, r, 0, r, grid[0], input, B, B);
        init_rect(r, N, 0, r, grid[1], input, 0, B);
        init_rect(E, r, 0, r, grid[2], input, 0, B);

        // west, mid, east
        init_rect(0, r, r, N, grid[3], input, B, 0);
        init_rect(r, N, r, N, grid[4], input, 0, 0);
        init_rect(E, r, r, N, grid[5], input, 0, 0);
    }
    // sw, south, se
    init_rect(0, r, E, r, grid[6], input, B, 0);
    init_rect(r, N, E, r, grid[7], input, 0, 0);
    init_rect(E, r, E, r, grid[8], input, 0, 0);

#undef B
#undef E
}
