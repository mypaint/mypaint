%module mypaintlib
%{
#include "mypaintlib.hpp"
%}

%include "std_vector.i"

namespace std {
   %template(IntVector) vector<int>;
   %template(DoubleVector) vector<double>;
}

typedef struct { int x, y, w, h; } Rect;

%include "surface.hpp"
%include "brush.hpp"
%include "mapping.hpp"

%include "python_brush.hpp"
%include "tiledsurface.hpp"

#ifdef HAVE_GEGL
%include "geglbackedsurface.hpp"
#endif

%include "pixops.hpp"
%include "colorring.hpp"
%include "colorchanger_wash.hpp"
%include "colorchanger_crossed_bowl.hpp"
%include "fastpng.hpp"

//from "gdkpixbuf2numpy.hpp"
PyObject * gdkpixbuf_numeric2numpy(PyObject * gdk_pixbuf_pixels_array);

%init %{
import_array();
%}

