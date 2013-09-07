#include "numpy/numpyconfig.h"
#ifdef NPY_1_7_API_VERSION
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#endif
#include "Python.h"
#include "numpy/arrayobject.h"

#include "mapping.hpp"
#include "surface.hpp"
#include "brush.hpp"
#include "python_brush.hpp"
#include "helpers2.hpp"
#include "tiledsurface.hpp"

#ifdef HAVE_GEGL
#include "geglbackedsurface.hpp"
#endif

#include "pixops.hpp"
#include "colorring.hpp"
#include "colorchanger_wash.hpp"
#include "colorchanger_crossed_bowl.hpp"
#include "gdkpixbuf2numpy.hpp"
#include "fastpng.hpp"
#include "fill.hpp"
