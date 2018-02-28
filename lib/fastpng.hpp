// Fast loading and saving using scalines
// Copyright (C) 2015-2018  The MyPaint Development Team
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


// Writes a PNG file progressively in strips

class ProgressivePNGWriter
{
public:
    ProgressivePNGWriter(PyObject *file,
                         const int w, const int h,
                         const bool has_alpha,
                         const bool save_srgb_chunks);
    PyObject *write(PyObject *arr);  // write a h*w*4 uint8 numpy array
    PyObject *close();   // finalize write
    ~ProgressivePNGWriter();
private:
    struct State;
    State *state;
};


// Load a file progressively as 8-bit RGBA, obtaining memory in NumPy
// array strips via a callback.

PyObject *
load_png_fast_progressive (char *filename,
                           PyObject *get_buffer_callback,
                           bool convert_to_srgb);

#endif //FASTPNG_HPP
