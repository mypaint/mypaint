/* Copyright 2004 Martin Renold <martinxyz@gmx.ch>
   Released under GPL.
*/

#include <gtk/gtk.h>
#include <string.h>
#include <math.h>
#include "wetpix.h"

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
          rgb_line[3*(x-x0) + 0] = r;
          rgb_line[3*(x-x0) + 1] = g;
          rgb_line[3*(x-x0) + 2] = b;
        }
      rgb_line += s->rowstride;
    }
}

void
surface_clear (Surface * s);
{
  int y;
  byte *rgb_line = s->rgb;

  /* clear rgb buffer to white */
  for (y = 0; y < s->h; y++)
    {
      memset (rgb_line, 255, width * 3);
      rgb_line += rgb_rowstride;
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
  double xx, yy, rr;

  r_fringe = b->radius + 1;
  x0 = floor (x - r_fringe);
  y0 = floor (y - r_fringe);
  x1 = ceil (x + r_fringe);
  y1 = ceil (y + r_fringe);
  if (x0 < 0) x0 = 0;
  if (y0 < 0) y0 = 0;
  if (x1 > s->w) x1 = s->w;
  if (y1 > s->h) y1 = s->h;

  rgb_line = s->rgb + y0 * s->rowstride;
  for (yp = y0; yp < y1; yp++)
    {
      yy = (yp + 0.5 - y);
      yy *= yy;
      for (xp = x0; xp < x1; xp++)
	{
	  xx = (xp + 0.5 - x);
	  xx *= xx;
	  rr = yy + xx;
	  if (rr < r * r)
	    press = pressure * 0.25;
	  else
	    press = -1;
}

void
brush_dab (byte *rgb,
           WetPix *paint,
           double x, double y,
           double r, double pressure,
           double strength)
{
  double r_fringe;
  int x0, y0;
  int x1, y1;
  WetPix *wet_line;
  int xp, yp;
  double xx, yy, rr;
  double eff_height;
  double press, contact;
  WetPixDbl wet_tmp, wet_tmp2;
  int maskstride = (layer->width + 3) >> 2;

  r_fringe = r + 1;
  x0 = floor (x - r_fringe);
  y0 = floor (y - r_fringe);
  x1 = ceil (x + r_fringe);
  y1 = ceil (y + r_fringe);

  wet_line = layer->buf + y0 * layer->rowstride;
  for (yp = y0; yp < y1; yp++)
    {
      yy = (yp + 0.5 - y);
      yy *= yy;
      for (xp = x0; xp < x1; xp++)
	{
	  xx = (xp + 0.5 - x);
	  xx *= xx;
	  rr = yy + xx;
	  if (rr < r * r)
	    press = pressure * 0.25;
	  else
	    press = -1;
	  eff_height = (wet_line[xp].h + wet_line[xp].w - 192) * (1.0 / 255);
	  contact = (press + eff_height) * 0.2;
	  if (contact > 0.5)
	    contact = 1 - 0.5 * exp (-2.0 * contact - 1);
	  if (contact > 0.0001)
	    {
	      int v;
	      double rnd = rand () * (1.0 / RAND_MAX);

	      v = wet_line[xp].rd;
	      wet_line[xp].rd = floor (v + (paint->rd * strength - v) * contact + rnd);
	      v = wet_line[xp].rw;
	      wet_line[xp].rw = floor (v + (paint->rw * strength - v) * contact + rnd);
	      v = wet_line[xp].gd;
	      wet_line[xp].gd = floor (v + (paint->gd * strength - v) * contact + rnd);
	      v = wet_line[xp].gw;
	      wet_line[xp].gw = floor (v + (paint->gw * strength - v) * contact + rnd);
	      v = wet_line[xp].bd;
	      wet_line[xp].bd = floor (v + (paint->bd * strength - v) * contact + rnd);
	      v = wet_line[xp].bw;
	      wet_line[xp].bw = floor (v + (paint->bw * strength - v) * contact + rnd);
	      v = wet_line[xp].w;
	      wet_line[xp].w = floor (v + (paint->w - v) * contact + rnd);

	      layer->mask[(yp >> 2) * maskstride + (xp >> 2)] = 1;
	    }
	}
      wet_line += layer->rowstride;
    }
}
