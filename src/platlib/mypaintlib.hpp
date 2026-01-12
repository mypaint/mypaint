// Inclusions needed to make the generated mypaintlib_wrap.cpp compile.

#include "common.hpp"

#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

// Main module, so don't use NO_IMPORT_ARRAY here
#include <pygobject.h>

#include "mapping.hpp"
#include "surface.hpp"
#include "brush.hpp"
#include "python_brush.hpp"
#include "helpers2.hpp"
#include "tiledsurface.hpp"

#include "pixops.hpp"
#include "colorring.hpp"
#include "colorchanger_wash.hpp"
#include "colorchanger_crossed_bowl.hpp"
#include "gdkpixbuf2numpy.hpp"
#include "fastpng.hpp"
#include "fill/fill_constants.hpp"
#include "fill/fill_common.hpp"
#include "fill/floodfill.hpp"
#include "fill/gap_detection.hpp"
#include "fill/blur.hpp"
#include "fill/morphology.hpp"
#include "brushsettings.hpp"
