// Workaround to bridge the gap between GdkPixbuf and NumPy
// Copyright (C) 2008-2014  Martin Renold
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

#ifndef GDKPIXBUF2NUMPY_HPP
#define GDKPIXBUF2NUMPY_HPP

#include <Python.h>


// Returns a NumPy array containing the pixel data of a GdkPixbuf. The returned
// array has dimensions HxWx3 if the pixbuf has no alpha channel, or HxWx4 if
// an alpha channel is present.

PyObject *gdkpixbuf_get_pixels_array(PyObject *pixbuf_pyobject);


#endif //GDKPIXBUF2NUMPY_HPP
