/* This file is part of MyPaint.
 * Copyright (C) 2007 by Martin Renold <martinxyz@gmx.ch>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License.
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY. See the COPYING file for more details.
 */

#ifndef __helpers_h__
#define __helpers_h__

#include <glib.h>
#include <assert.h>

gdouble rand_gauss (GRand * rng);

// MAX, MIN, ABS, CLAMP defined in gmacros.h
#define ROUND(x) ((int) ((x) + 0.5))
#define SIGN(x) ((x)>0?1:(-1))
#define SQR(x) ((x)*(x))

#define MAX3(a, b, c) ((a)>(b)?MAX((a),(c)):MAX((b),(c)))
#define MIN3(a, b, c) ((a)<(b)?MIN((a),(c)):MIN((b),(c)))

void rgb_to_hsv_int (gint *red /*h*/, gint *green /*s*/, gint *blue /*v*/);
void hsv_to_rgb_int (gint *hue /*r*/, gint *saturation /*g*/, gint *value /*b*/);
void hsl_to_rgb_int (gint *hue, gint *saturation, gint *lightness);

void rgb_to_hsv_float (float *r_ /*h*/, float *g_ /*s*/, float *b_ /*v*/);
void hsv_to_rgb_float (float *h_, float *s_, float *v_);
void hsl_to_rgb_float (float *h_, float *s_, float *l_);
void rgb_to_hsl_float (float *r_, float *g_, float *b_);


typedef struct { int x, y, w, h; } Rect;
void ExpandRectToIncludePoint(Rect * r, int x, int y);

#endif
