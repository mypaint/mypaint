/* This file is part of MyPaint.
 * Copyright (C) 2019 by the MyPaint Development Team.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */

#ifndef BLUR_HPP
#define BLUR_HPP

#include "fill_common.hpp"
#include "blur_swig.hpp"

/*
  Holds data and allocated memory used to perform per-tile gaussian blur.

  The blur is performed in two passes. First the full input is blurred
  horizontally, writing the output to an intermediate array. Secondly the
  intermediate array is blurred vertically, writing the output into a new tile.
*/
class GaussBlurrer
{
  public:
    explicit GaussBlurrer(int radius);
    ~GaussBlurrer();
    PyObject* blur(bool can_update, GridVector input);

  private:
    // Read in-data from the tiles of input to the input_full array
    void initiate(bool can_update, GridVector input);
    // Predicates checking the state of input_full, relative
    // to the tiles in the most recent call to initiate
    bool input_is_fully_opaque();
    bool input_is_fully_transparent();
    // Blur factors used to calculate the value of every blurred pixel
    // based on its horizontal
    const std::vector<fix15_short_t> factors;
    const int radius;
    chan_t** input_full;
    chan_t** input_vertical;
};


#endif //BLUR_HPP
