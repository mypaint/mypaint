// Fast loading and saving using scalines
// Copyright (C) 2015  Andrew Chadwick
//
// This program is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
//
// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
// more details.
//
// You should have received a copy of the GNU General Public License along with
// this program; if not, write to the Free Software Foundation, Inc., 51
// Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

#ifndef FASTPNG_HPP
#define FASTPNG_HPP

#include <Python.h>


// Save a PNG file progressively, obtaining strips to write using a
// Python generator.

PyObject *
save_png_fast_progressive (char *filename,
                           int w, int h,
                           bool has_alpha,
                           PyObject *data_generator,
                           bool save_srgb_chunks);


// Load a file progressively as 8-bit RGBA, obtaining memory in NumPy
// array strips via a callback.

PyObject *
load_png_fast_progressive (char *filename,
                           PyObject *get_buffer_callback,
                           bool convert_to_srgb);

#endif //FASTPNG_HPP
