/* This file is part of MyPaint.
 * Copyright (C) 2019 by the MyPaint Development Team.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */

#ifndef FILL_CONSTANTS_HPP
#define FILL_CONSTANTS_HPP

#include "fill_common.hpp"

/*
  Constant tiles, mostly used for pointer comparison
  but also read from into nine-grid data sections.

  The constant opaque tile is filled with the fix15
  representation of 1, whereas the constant transparent
  tile is filled with 0.

  Both tiles should be treated as read-only, although
  this is not explicitly enforced.
*/
class ConstTiles
{
  public:
    static PyObject* ALPHA_OPAQUE();
    static PyObject* ALPHA_TRANSPARENT();

  private:
    static void init();
    static PyObject* _ALPHA_OPAQUE;
    static PyObject* _ALPHA_TRANSPARENT;
};

#endif //FILL_CONSTANTS_HPP
