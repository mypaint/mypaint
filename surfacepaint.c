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
  s = g_new (Surface, 1);

  s->rowstride = w * 3;
  s->rowstride = (s->rowstride + 3) & -4; /* align to 4-byte boundary */

  s->rgb = g_new(byte, h*s->rowstride);
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
  int x, y;
  byte *rgb_line;

  rgb_line = s->rgb;
  for (y = 0; y < s->h; y++)
    {
      for (x = 0; x < s->w; x++) 
        {
          int r, g, b;
          r = x % 256;
          g = y % 256;
          b = (x*x+y*y) % 256;
          rgb_line[3*x + 0] = r;
          rgb_line[3*x + 1] = g;
          rgb_line[3*x + 2] = b;
        }
      rgb_line += s->rowstride;
    }
}

void
surface_clear (Surface * s)
{
  int y;
  byte *rgb_line = s->rgb;

  /* clear rgb buffer to white */
  for (y = 0; y < s->h; y++)
    {
      memset (rgb_line, 255, s->w * 3);
      rgb_line += s->rowstride;
    }
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
  byte *rgb_line;
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
  if (opaque < 0) opaque = 0;
  if (opaque > 256) opaque = 256;
  c[0] = b->color[0];
  c[1] = b->color[1];
  c[2] = b->color[2];
  radius2 = sqr(b->radius);

  rgb_line = s->rgb + y0 * s->rowstride;
  for (yp = y0; yp < y1; yp++)
    {
      yy = (yp + 0.5 - y);
      yy *= yy;
      rgb = rgb_line + 3*x0;
      for (xp = x0; xp < x1; xp++)
	{
	  xx = (xp + 0.5 - x);
	  xx *= xx;
	  rr = yy + xx;
	  if (rr < radius2) {
            rgb[0] = (opaque*c[0] + (256-opaque)*rgb[0]) / 256;
            rgb[1] = (opaque*c[1] + (256-opaque)*rgb[1]) / 256;
            rgb[2] = (opaque*c[2] + (256-opaque)*rgb[2]) / 256;
          }
          rgb += 3;
        }
      rgb_line += s->rowstride;
    }
}

