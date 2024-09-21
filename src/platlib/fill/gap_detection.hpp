/* This file is part of MyPaint.
 * Copyright (C) 2018 by the MyPaint Development Team.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */

#ifndef GAP_DETECTION_HPP
#define GAP_DETECTION_HPP

#include "fill_common.hpp"

/*
  Distance data bucket for gap closing
*/
class DistanceBucket
{
  public:
    explicit DistanceBucket(int distance);
    ~DistanceBucket();
    const int distance;
    chan_t** input;
};

/*
   Search the given nine-grid of flooded alpha tiles for
   gaps up to a certain length, defined by the DistanceBucket,
   writing the lengths found to the given distance tile
   Returns true if any gaps were found.
*/
bool find_gaps(
    DistanceBucket& bucket, PyObject* gap_output, PyObject* src_mid,
    PyObject* src_n, PyObject* src_e, PyObject* src_s, PyObject* src_w,
    PyObject* src_ne, PyObject* src_se, PyObject* src_sw, PyObject* src_nw);

#endif
