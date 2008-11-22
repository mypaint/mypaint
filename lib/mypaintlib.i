%module mypaintlib
%{
#include "mypaintlib.hpp"
%}

typedef struct { int x, y, w, h; } Rect;

%include "surface.hpp"
%include "brush.hpp"
%include "tiledsurface.hpp"
%include "colorselector.hpp"

//from "gdkpixbuf2numpy.hpp"
PyObject * gdkpixbuf2numpy(PyObject * gdk_pixbuf_pixels_array);

%init %{
import_array();
%}

