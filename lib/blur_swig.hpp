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
  Apply a gaussian blur to the input tiles, adding the 
  resulting output tiles to the given dictionary.
*/
void blur(
    int radius, // Radius to grow (if > 0) or shrink (if < 0)
    PyObject* blurred, // Dictionary holding the result of the operation
    PyObject* tiles, // Input tiles, NxNx1 uint16 numpy arrays
    PyObject* strands // Strands of contiguous tile coordinates
    );

#endif //BLUR_SWIG_HPP
