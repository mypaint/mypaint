/* Copyright 2004 Martin Renold <martinxyz@gmx.ch>
   Released as public domain.
*/

#include <gtk/gtk.h>
#include <string.h>
#include <math.h>
#include "surface.h"

Surface *
new_surface (int w, int h)
{
  Surface * s;
  if (w != SIZE || h != SIZE) {
    g_print ("Only %d*%d supported right now\n", SIZE, SIZE);
    return NULL;
  }
  s = g_new0 (Surface, 1);

  /*
  s->rowstride = w * 3;
  s->rowstride = (s->rowstride + 3) & -4; */ /* align to 4-byte boundary */

  /* s->rgb = g_new(guchar, h*s->rowstride); */
  s->rgb = g_new0 (guchar, 3*w*h);

  s->w = w;
  s->h = h;
  return s;
}

void
free_surface (Surface * s)
{
  g_free (s->rgb);
  g_free (s);
}

void 
surface_renderpattern (Surface * s)
{
  int x, y, r, g, b;
  guchar *rgb;

  for (y = 0; y < s->h; y++) {
    for (x = 0; x < s->w; x++) {
      rgb = PixelXY(s, x, y);
      r = x % 256;
      g = y % 256;
      b = (x*x+y*y) % 256;
      rgb[3*x + 0] = r;
      rgb[3*x + 1] = g;
      rgb[3*x + 2] = b;
    }
  }
}

void
surface_clear (Surface * s)
{
  // clear rgb buffer to white
  memset (s->rgb, 255, s->w*s->h*3);
  if (s->widget) gtk_widget_queue_draw (s->widget);
}

void
surface_render (Surface * s,
                guchar * dst, int rowstride,
                int x0, int y0,
                int w, int h)
{
  // could be optimized much, important if big brush is used
  int x, y;
  //g_print("%d %d\n", w, h);
  guchar * rgb_line = dst;
  guchar * rgb_dst;
  guchar * rgb_src;
  for (y = y0; y < y0 + h; y++) {
    rgb_dst = rgb_line;
    for (x = x0; x < x0 + w; x++) {
      rgb_src = PixelXY(s, x, y);
      rgb_dst[0] = rgb_src[0];
      rgb_dst[1] = rgb_src[1];
      rgb_dst[2] = rgb_src[2];
      rgb_dst += 3;
    }
    rgb_line += rowstride;
  }
}

