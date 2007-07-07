/* This file is part of MyPaint.
 * Copyright (C) 2007 by Martin Renold <martinxyz@gmx.ch>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License.
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY. See the COPYING file for more details.
 */

#ifndef __GESTURES_H__
#define __GESTURES_H__

enum {
  GESTURE_NONE = 0,
  GESTURE_COLORSELECT
};


int detect_gestures (double time, double x, double y, double pressure);

#endif
