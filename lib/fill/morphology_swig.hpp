/* This file is part of MyPaint.
 * Copyright (C) 2019 by the MyPaint Development Team.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */

#ifndef MORPHOLOGY_SWIG_HPP
#define MORPHOLOGY_SWIG_HPP

#include "fill_common.hpp"

/*
   Perform a dilation or erosion using the given input tiles
   and strands of vertically contiguous coordinates, placing
   the result in the given coord->tile dictionary.
*/
void morph(
    int offset, // Radius to grow (if > 0) or shrink (if < 0)
    PyObject* morphed, // Dictionary holding the result of the operation
    PyObject* tiles, // Input tiles, NxNx1 uint16 numpy arrays
    PyObject* strands, // Strands of contiguous tile coordinates
    Controller& status_controller // cancellation and status data
    );

#endif //MORPHOLOGY_SWIG_HPP
