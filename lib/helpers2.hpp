
#ifndef HELPERS2_HPP
#define HELPERS2_HPP

// Making the helpers of brushlib also bound by Python
#include "helpers.c"
#include <glib.h>

// Special HSV -> RGB converter for use with the color selector classes
// Takes values in the range [ 0.0 , 1.0 ]
// Gives values in the range [ 0.0 , 255.0 ]
void hsv_to_rgb_range_one(float *h_, float *s_, float *v_)
{
  gint i;
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

  gdouble hue = h;

  if( hue == 1.0 )
    hue = 0.0;
  else
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

  *h_ = r*255.0f;
  *s_ = g*255.0f;
  *v_ = b*255.0f;
}

typedef struct { int x, y, w, h; } Rect;
// originally from my mass project (mass.sourceforge.net)
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

#endif //HELPERS2_HPP
