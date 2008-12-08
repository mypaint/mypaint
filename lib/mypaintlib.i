%module mypaintlib
%{
#include "mypaintlib.hpp"
%}

typedef struct { int x, y, w, h; } Rect;

%include "../brushlib/surface.hpp"
%include "../brushlib/brush.hpp"
%include "tiledsurface.hpp"
%include "colorselector.hpp"
%include "colorchanger.hpp"

//from "gdkpixbuf2numpy.hpp"
PyObject * gdkpixbuf2numpy(PyObject * gdk_pixbuf_pixels_array);

%init %{
import_array();
%}

