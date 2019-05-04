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
  Holds data and allocated space used to perform
  per-tile gaussian blur.
*/
class BlurBucket
{
  public:
    explicit BlurBucket(int radius);
    ~BlurBucket();
    PyObject* blur(bool can_update, GridVector input);

  private:
    void initiate(bool can_update, GridVector input);
    bool input_fully_opaque();
    bool input_fully_transparent();
    const std::vector<fix15_short_t> factors;
    const int radius;
    chan_t** input_full;
    chan_t** input_vert;
    chan_t output[N][N];
};


#endif //BLUR_HPP
