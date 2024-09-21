/* This file is part of MyPaint.
 * Copyright (C) 2019 by the MyPaint Development Team.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */

#include "fill_constants.hpp"

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

