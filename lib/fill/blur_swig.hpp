/* This file is part of MyPaint.
 * Copyright (C) 2019 by the MyPaint Development Team.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */

#ifndef BLUR_SWIG_HPP
#define BLUR_SWIG_HPP

#include "fill_common.hpp"

/*
  Apply a gaussian blur to the input tiles, adding the resulting
  new blurred tiles to the given dictionary. Tiles in the output
  dictionary are not guaranteed to be unique; the constant fully
  opaque tile will be used wherever possible.
*/
void blur(
    int radius, // Nominal blur radius (real radius may be larger or smaller)
    PyObject* blurred, // Dictionary holding the result of the operation
    PyObject* tiles, // Input tiles, NxNx1 uint16 numpy arrays
    PyObject* strands, // List of lists of vertically contiguous coordinates
    Controller& status_controller // cancellation and status data
    );

#endif //BLUR_SWIG_HPP
