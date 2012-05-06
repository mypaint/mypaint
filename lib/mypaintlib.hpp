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
