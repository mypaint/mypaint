/* This file is part of MyPaint.
 * Copyright (C) 2014 by Andrew Chadwick <a.t.chadwick@gmail.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */

// Common stuff

#ifndef COMMON_HPP
#define COMMON_HPP

// Make the "HEAVY_DEBUG" macro readable from python

#ifdef HEAVY_DEBUG
const bool heavy_debug = true;
#else
const bool heavy_debug = false;
#endif

// The mypaintlib stuff is split over multiple source files, so we need to
// define a shared name to use for the static NumPy API.
// Ref: http://docs.scipy.org/doc/numpy/reference/c-api.array.html

#define PY_ARRAY_UNIQUE_SYMBOL mypaintlib_Array_API

// If you're using NumPy C API stuff in a separately compiled submodule, it
// must be imported like this in each implementation file.
//
//  #include "common.hpp"
//  #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
//  #define NO_IMPORT_ARRAY
//  #include <numpy/arrayobject.h>
//
// Ref: http://docs.scipy.org/doc/numpy-dev/reference/c-api.deprecations.html

#endif //COMMON_HPP
