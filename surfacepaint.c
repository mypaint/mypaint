/* Copyright 2004 Martin Renold <martinxyz@gmx.ch>
   Released under GPL.
*/

#include <gtk/gtk.h>
#include <string.h>
#include <math.h>
#include "surfacepaint.h"

Surface *
new_surface (int w, int h)
{
  Surface * s;
  if (w != SIZE || h != SIZE) 
    {
      g_print ("Only %d*%d supported right now\n", SIZE, SIZE);
      return NULL;
    }
  s = g_new (Surface, 1);

  /*
  s->rowstride = w * 3;
  s->rowstride = (s->rowstride + 3) & -4; */ /* align to 4-byte boundary */

  /* s->rgb = g_new(byte, h*s->rowstride); */
  s->rgb = g_new(byte, 3*w*h);

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
  byte *rgb;

  for (y = 0; y < s->h; y++)
    {
      for (x = 0; x < s->w; x++) 
        {
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
  /* clear rgb buffer to white */
  memset (s->rgb, 255, s->w*s->h*3);
}

void
surface_draw (Surface * s,
              double x, double y,
              Brush * b)
{
  double r_fringe;
  int x0, y0;
  int x1, y1;
  int xp, yp;
  byte *rgb;
  double xx, yy, rr;
  double radius2;
  int opaque;
  byte c[3];

  r_fringe = b->radius + 1;
  x0 = floor (x - r_fringe);
  y0 = floor (y - r_fringe);
  x1 = ceil (x + r_fringe);
  y1 = ceil (y + r_fringe);
  if (x0 < 0) x0 = 0;
  if (y0 < 0) y0 = 0;
  if (x1 > s->w) x1 = s->w;
  if (y1 > s->h) y1 = s->h;
  rr = sqr(b->radius);
  opaque = floor(b->opaque * 256 + 0.5);
  if (opaque <= 0) return;
  if (opaque > 256) opaque = 256;
  c[0] = b->color[0];
  c[1] = b->color[1];
  c[2] = b->color[2];
  radius2 = sqr(b->radius);

  for (yp = y0; yp < y1; yp++)
    {
      yy = (yp + 0.5 - y);
      yy *= yy;
      for (xp = x0; xp < x1; xp++)
	{
	  xx = (xp + 0.5 - x);
	  xx *= xx;
	  rr = yy + xx;
          rgb = PixelXY(s, xp, yp);
	  if (rr < radius2) {
            rgb[0] = (opaque*c[0] + (256-opaque)*rgb[0]) / 256;
            rgb[1] = (opaque*c[1] + (256-opaque)*rgb[1]) / 256;
            rgb[2] = (opaque*c[2] + (256-opaque)*rgb[2]) / 256;
          }
          rgb += 3;
        }
    }
}

void
surface_render (Surface * s,
                byte * dst, int rowstride,
                int x0, int y0,
                int w, int h)
{
  /* could be optimized much, important if big brush is used */
  int x, y;
  /*g_print("%d %d\n", w, h);*/
  byte * rgb_line = dst;
  byte * rgb_dst;
  byte * rgb_src;
  for (y = y0; y < y0 + h; y++) 
    {
      rgb_dst = rgb_line;
      for (x = x0; x < x0 + w; x++)
        {
          rgb_src = PixelXY(s, x, y);
          rgb_dst[0] = rgb_src[0];
          rgb_dst[1] = rgb_src[1];
          rgb_dst[2] = rgb_src[2];
          rgb_dst += 3;
        }
      rgb_line += rowstride;
    }
}

