/* This file is part of MyPaint.
 * Copyright (C) 2007 by Martin Renold <martinxyz@gmx.ch>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License.
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY. See the COPYING file for more details.
 */

#include "helpers.h"

typedef struct {
  int dtime;
  float x, y;
  float pressure;
} StrokeEvent;

GString* event_array_to_string (GArray * ea);
GArray* string_to_event_array (GString * bs);
