// Workaround to bridge the gap between GdkPixbuf and NumPy
// Copyright (C) 1998-2003  James Henstridge
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

#include "gdkpixbuf2numpy.hpp"

#include "common.hpp"

#include <gtk/gtk.h>

#define NO_IMPORT_PYGOBJECT
#include <pygobject.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>


// gdk_pixbuf_get_pixels_array() isn't supported in GI-era Python GTK: it was
// always a pygtk convenience function.

// Near-verbatim lift of gdk_pixbuf_get_pixels_array() from gdkpixbuf.override
//
// Originally written by James Henstridge, and published under the terms of the
// GNU Lesser General Public License, version 2.1.

PyObject *
gdkpixbuf_get_pixels_array(PyObject *pixbuf_pyobject)
{
    GdkPixbuf *pixbuf = GDK_PIXBUF(((PyGObject *)pixbuf_pyobject)->obj);
    PyArrayObject *array;
    npy_intp dims[3] = { 0, 0, 3 };

    dims[0] = gdk_pixbuf_get_height(pixbuf);
    dims[1] = gdk_pixbuf_get_width(pixbuf);

    if (gdk_pixbuf_get_has_alpha(pixbuf))
        dims[2] = 4;
    guchar *pixels = gdk_pixbuf_get_pixels(pixbuf);

    array = (PyArrayObject *)PyArray_SimpleNewFromData(3, dims, NPY_UBYTE,
                                                       pixels);

    if (array == NULL)
        return NULL;

    PyArray_STRIDES(array)[0] = gdk_pixbuf_get_rowstride(pixbuf);
    // the array holds a ref to the pixbuf pixels through this wrapper
    Py_INCREF(pixbuf_pyobject);

#ifdef NPY_1_7_API_VERSION
    PyArray_SetBaseObject(array, (PyObject *)pixbuf_pyobject);
#else
    array->base = (PyObject *)pixbuf_pyobject;
#endif

    return PyArray_Return(array);
}
