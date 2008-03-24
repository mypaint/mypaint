/* This file is part of MyPaint.
 * Copyright (C) 2007 by Martin Renold <martinxyz@gmx.ch>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License.
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY. See the COPYING file for more details.
 *
 * This file contains some modified code from the GIMP:
 * Copyright (C) 1995-1997 Peter Mattis and Spencer Kimball
 * Adapted 2007 by Martin Renold to fit into MyPaint.
 * Also fixed the hue range in gimp_hsl_to_rgb_int().
 */

#include "helpers.hpp"
#include <glib.h>
#include <math.h>

// stolen from the gimp (noisify.c)

/*
 * Return a Gaussian (aka normal) random variable.
 *
 * Adapted from ppmforge.c, which is part of PBMPLUS.
 * The algorithm comes from:
 * 'The Science Of Fractal Images'. Peitgen, H.-O., and Saupe, D. eds.
 * Springer Verlag, New York, 1988.
 *
 * It would probably be better to use another algorithm, such as that
 * in Knuth
 */
gdouble rand_gauss (GRand * rng)
{
  gint i;
  gdouble sum = 0.0;

  for (i = 0; i < 4; i++) {
    sum += g_rand_int_range (rng, 0, 0x7FFF);
  }

  return sum * 5.28596089837e-5 - 3.46410161514;
}

// stolen from the gimp (gimpcolorspace.c)

/*  gint functions  */

/**
 * gimp_rgb_to_hsv_int:
 * @red: The red channel value, returns the Hue channel
 * @green: The green channel value, returns the Saturation channel
 * @blue: The blue channel value, returns the Value channel
 *
 * The arguments are pointers to int representing channel values in
 * the RGB colorspace, and the values pointed to are all in the range
 * [0, 255].
 *
 * The function changes the arguments to point to the HSV value
 * corresponding, with the returned values in the following
 * ranges: H [0, 360], S [0, 255], V [0, 255].
 **/
void
rgb_to_hsv_int (gint *red,
                gint *green,
                gint *blue)
{
  gdouble  r, g, b;
  gdouble  h, s, v;
  gint     min;
  gdouble  delta;

  r = *red;
  g = *green;
  b = *blue;

  if (r > g)
    {
      v = MAX (r, b);
      min = MIN (g, b);
    }
  else
    {
      v = MAX (g, b);
      min = MIN (r, b);
    }

  delta = v - min;

  if (v == 0.0)
    s = 0.0;
  else
    s = delta / v;

  if (s == 0.0)
    h = 0.0;
  else
    {
      if (r == v)
	h = 60.0 * (g - b) / delta;
      else if (g == v)
	h = 120 + 60.0 * (b - r) / delta;
      else
	h = 240 + 60.0 * (r - g) / delta;

      if (h < 0.0)
	h += 360.0;
      if (h > 360.0)
	h -= 360.0;
    }

  *red   = ROUND (h);
  *green = ROUND (s * 255.0);
  *blue  = ROUND (v);
}

/**
 * gimp_hsv_to_rgb_int:
 * @hue: The hue channel, returns the red channel
 * @saturation: The saturation channel, returns the green channel
 * @value: The value channel, returns the blue channel
 *
 * The arguments are pointers to int, with the values pointed to in the
 * following ranges:  H [0, 360], S [0, 255], V [0, 255].
 *
 * The function changes the arguments to point to the RGB value
 * corresponding, with the returned values all in the range [0, 255].
 **/
void
hsv_to_rgb_int (gint *hue,
                gint *saturation,
                gint *value)
{
  gdouble h, s, v, h_temp;
  gdouble f, p, q, t;
  gint i;

  if (*saturation == 0)
    {
      *hue        = *value;
      *saturation = *value;
      *value      = *value;
    }
  else
    {
      h = *hue;
      s = *saturation / 255.0;
      v = *value      / 255.0;

      if (h == 360)
         h_temp = 0;
      else
         h_temp = h;

      h_temp = h_temp / 60.0;
      i = floor (h_temp);
      f = h_temp - i;
      p = v * (1.0 - s);
      q = v * (1.0 - (s * f));
      t = v * (1.0 - (s * (1.0 - f)));

      switch (i)
	{
	case 0:
	  *hue        = ROUND (v * 255.0);
	  *saturation = ROUND (t * 255.0);
	  *value      = ROUND (p * 255.0);
	  break;

	case 1:
	  *hue        = ROUND (q * 255.0);
	  *saturation = ROUND (v * 255.0);
	  *value      = ROUND (p * 255.0);
	  break;

	case 2:
	  *hue        = ROUND (p * 255.0);
	  *saturation = ROUND (v * 255.0);
	  *value      = ROUND (t * 255.0);
	  break;

	case 3:
	  *hue        = ROUND (p * 255.0);
	  *saturation = ROUND (q * 255.0);
	  *value      = ROUND (v * 255.0);
	  break;

	case 4:
	  *hue        = ROUND (t * 255.0);
	  *saturation = ROUND (p * 255.0);
	  *value      = ROUND (v * 255.0);
	  break;

	case 5:
	  *hue        = ROUND (v * 255.0);
	  *saturation = ROUND (p * 255.0);
	  *value      = ROUND (q * 255.0);
	  break;
	}
    }
}


static gint
gimp_hsl_value_int (gdouble n1,
                    gdouble n2,
                    gdouble hue)
{
  gdouble value;

  if (hue > 360)
    hue -= 360;
  else if (hue < 0)
    hue += 360;

  if (hue < 60.0)
    value = n1 + (n2 - n1) * (hue / 60.0);
  else if (hue < 180.0)
    value = n2;
  else if (hue < 240)
    value = n1 + (n2 - n1) * ((240 - hue) / 60.0);
  else
    value = n1;

  /*
  if (hue < 1.0)
    val = n1 + (n2 - n1) * hue;
  else if (hue < 3.0)
    val = n2;
  else if (hue < 4.0)
    val = n1 + (n2 - n1) * (4.0 - hue);
  else
    val = n1;
  */

  return ROUND (value * 255.0);
}

/**
 * gimp_hsl_to_rgb_int:
 * @hue: Hue channel, returns Red channel
 * @saturation: Saturation channel, returns Green channel
 * @lightness: Lightness channel, returns Blue channel
 *
 * The arguments are pointers to int, with the values pointed to in the
 * following ranges:  H [0, 360], L [0, 255], S [0, 255].
 *
 * The function changes the arguments to point to the RGB value
 * corresponding, with the returned values all in the range [0, 255].
 **/
void
hsl_to_rgb_int (gint *hue,
                gint *saturation,
                gint *lightness)
{
  gdouble h, s, l;

  h = *hue;
  s = *saturation;
  l = *lightness;

  if (s == 0)
    {
      /*  achromatic case  */
      *hue        = l;
      *lightness  = l;
      *saturation = l;
    }
  else
    {
      gdouble m1, m2;

      if (l < 128)
        m2 = (l * (255 + s)) / 65025.0;
      else
        m2 = (l + s - (l * s) / 255.0) / 255.0;

      m1 = (l / 127.5) - m2;

      /*  chromatic case  */
      *hue        = gimp_hsl_value_int (m1, m2, h + 120);
      *saturation = gimp_hsl_value_int (m1, m2, h);
      *lightness  = gimp_hsl_value_int (m1, m2, h - 120);
    }
}



// (from gimp_rgb_to_hsv)
void
rgb_to_hsv_float (float *r_, float *g_, float *b_)
{
  float max, min, delta;
  float h, s, v;
  float r, g, b;

  h = 0.0; // silence gcc warning

  r = *r_;
  g = *g_;
  b = *b_;

  r = CLAMP(r, 0.0, 1.0);
  g = CLAMP(g, 0.0, 1.0);
  b = CLAMP(b, 0.0, 1.0);

  max = MAX3(r, g, b);
  min = MIN3(r, g, b);

  v = max;
  delta = max - min;

  if (delta > 0.0001)
    {
      s = delta / max;

      if (r == max)
        {
          h = (g - b) / delta;
          if (h < 0.0)
            h += 6.0;
        }
      else if (g == max)
        {
          h = 2.0 + (b - r) / delta;
        }
      else if (b == max)
        {
          h = 4.0 + (r - g) / delta;
        }

      h /= 6.0;
    }
  else
    {
      s = 0.0;
      h = 0.0;
    }

  *r_ = h;
  *g_ = s;
  *b_ = v;
}

// (from gimp_hsv_to_rgb)
void
hsv_to_rgb_float (float *h_, float *s_, float *v_)
{
  gint    i;
  gdouble f, w, q, t;
  float h, s, v;
  float r, g, b;
  r = g = b = 0.0; // silence gcc warning

  h = *h_;
  s = *s_;
  v = *v_;

  h = h - floor(h);
  s = CLAMP(s, 0.0, 1.0);
  v = CLAMP(v, 0.0, 1.0);

  gdouble hue;

  if (s == 0.0)
    {
      r = v;
      g = v;
      b = v;
    }
  else
    {
      hue = h;

      if (hue == 1.0)
        hue = 0.0;

      hue *= 6.0;

      i = (gint) hue;
      f = hue - i;
      w = v * (1.0 - s);
      q = v * (1.0 - (s * f));
      t = v * (1.0 - (s * (1.0 - f)));

      switch (i)
        {
        case 0:
          r = v;
          g = t;
          b = w;
          break;
        case 1:
          r = q;
          g = v;
          b = w;
          break;
        case 2:
          r = w;
          g = v;
          b = t;
          break;
        case 3:
          r = w;
          g = q;
          b = v;
          break;
        case 4:
          r = t;
          g = w;
          b = v;
          break;
        case 5:
          r = v;
          g = w;
          b = q;
          break;
        }
    }

  *h_ = r;
  *s_ = g;
  *v_ = b;
}

// gimp_rgb_to_hsl:
void
rgb_to_hsl_float (float *r_, float *g_, float *b_)
{
  gdouble max, min, delta;

  float h, s, l;
  float r, g, b;

  // silence gcc warnings
  h=0;

  r = *r_;
  g = *g_;
  b = *b_;

  r = CLAMP(r, 0.0, 1.0);
  g = CLAMP(g, 0.0, 1.0);
  b = CLAMP(b, 0.0, 1.0);

  max = MAX3(r, g, b);
  min = MIN3(r, g, b);

  l = (max + min) / 2.0;

  if (max == min)
    {
      s = 0.0;
      h = 0.0; //GIMP_HSL_UNDEFINED;
    }
  else
    {
      if (l <= 0.5)
        s = (max - min) / (max + min);
      else
        s = (max - min) / (2.0 - max - min);

      delta = max - min;

      if (delta == 0.0)
        delta = 1.0;

      if (r == max)
        {
          h = (g - b) / delta;
        }
      else if (g == max)
        {
          h = 2.0 + (b - r) / delta;
        }
      else if (b == max)
        {
          h = 4.0 + (r - g) / delta;
        }

      h /= 6.0;

      if (h < 0.0)
        h += 1.0;
    }

  *r_ = h;
  *g_ = s;
  *b_ = l;
}

static double
hsl_value (gdouble n1,
           gdouble n2,
           gdouble hue)
{
  gdouble val;

  if (hue > 6.0)
    hue -= 6.0;
  else if (hue < 0.0)
    hue += 6.0;

  if (hue < 1.0)
    val = n1 + (n2 - n1) * hue;
  else if (hue < 3.0)
    val = n2;
  else if (hue < 4.0)
    val = n1 + (n2 - n1) * (4.0 - hue);
  else
    val = n1;

  return val;
}


/**
 * gimp_hsl_to_rgb:
 * @hsl: A color value in the HSL colorspace
 * @rgb: The value converted to a value in the RGB colorspace
 *
 * Convert a HSL color value to an RGB color value.
 **/
void
hsl_to_rgb_float (float *h_, float *s_, float *l_)
{
  float h, s, l;
  float r, g, b;

  h = *h_;
  s = *s_;
  l = *l_;

  h = h - floor(h);
  s = CLAMP(s, 0.0, 1.0);
  l = CLAMP(l, 0.0, 1.0);

  if (s == 0)
    {
      /*  achromatic case  */
      r = l;
      g = l;
      b = l;
    }
  else
    {
      gdouble m1, m2;

      if (l <= 0.5)
        m2 = l * (1.0 + s);
      else
        m2 = l + s - l * s;

      m1 = 2.0 * l - m2;

      r = hsl_value (m1, m2, h * 6.0 + 2.0);
      g = hsl_value (m1, m2, h * 6.0);
      b = hsl_value (m1, m2, h * 6.0 - 2.0);
    }

  *h_ = r;
  *s_ = g;
  *l_ = b;
}

// tested, copied from my mass project
void ExpandRectToIncludePoint(Rect * r, int x, int y) 
{
  if (r->w == 0) {
    r->w = 1; r->h = 1;
    r->x = x; r->y = y;
  } else {
    if (x < r->x) { r->w += r->x-x; r->x = x; } else
    if (x >= r->x+r->w) { r->w = x - r->x + 1; }

    if (y < r->y) { r->h += r->y-y; r->y = y; } else
    if (y >= r->y+r->h) { r->h = y - r->y + 1; }
  }
}
