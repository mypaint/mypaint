#include "helpers.h"
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
gdouble gauss_noise (void)
{
  gint i;
  gdouble sum = 0.0;

  for (i = 0; i < 4; i++)
    sum += g_random_int_range (0, 0x7FFF);

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
gimp_rgb_to_hsv_int (gint *red,
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
gimp_hsv_to_rgb_int (gint *hue,
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
