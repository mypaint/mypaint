#include <stdio.h>
#include <string.h>
#include <glib.h>
#include <math.h>
#include "brush.h"

Brush * brush_create ()
{
  Brush * b;
  b = g_new0 (Brush, 1);
  b->radius = 4.0;
  b->spacing = 0.5;
  b->opaque = 1.0;
  b->queue_draw_widget = NULL;
  return b;
}

Brush * brush_create_copy (Brush * old_b)
{
  Brush * b;
  b = g_new0 (Brush, 1);
  memcpy (b, old_b, sizeof(Brush));
  return b;
}

void brush_free (Brush * b)
{
  g_free (b);
}

void brush_reset (Brush * b)
{
  b->time = 0;
}

void brush_mutate (Brush * b)
{
  int i;
  for (i=0; i<F_WEIGHTS; i++) {
    /*
    if (g_random_int_range(0, 10) == 0) {
      b->variations[i] *= g_random_double_range(0.5, 1.0/0.5);
      }*/
    b->weights[i] += g_random_double_range(-b->variations[i], b->variations[i]);
  }
}

void brush_dab (Brush * b, Surface * s) {
  double r_fringe;
  int x0, y0;
  int x1, y1;
  int xp, yp;
  guchar *rgb;
  double xx, yy, rr;
  double radius2;
  int opaque;
  guchar c[3];
  if (!s) return;

  r_fringe = b->radius + 1;
  x0 = floor (b->x - r_fringe);
  y0 = floor (b->y - r_fringe);
  x1 = ceil (b->x + r_fringe);
  y1 = ceil (b->y + r_fringe);
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

  for (yp = y0; yp < y1; yp++) {
    yy = (yp + 0.5 - b->y);
    yy *= yy;
    for (xp = x0; xp < x1; xp++) {
      xx = (xp + 0.5 - b->x);
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
  
  if (b->queue_draw_widget) {
    gtk_widget_queue_draw_area (b->queue_draw_widget,
                                floor(b->x - (b->radius+1)),
                                floor(b->y - (b->radius+1)),
                                /* FIXME: think about it exactly */
                                ceil (2*(b->radius+1)),
                                ceil (2*(b->radius+1))
                                );
  }
}

// high-level part of before each dab
void brush_prepare_dab (Brush * b)
{
  float speed;
  float noise;
  float sum;
  noise = g_random_double (); /* [0..1) */
  speed = sqrt(sqr(b->dx) + sqr(b->dy))/b->dtime;
  b->opaque = b->pressure / 8.0;
  //b->radius = 0.1 + 60 * b->pressure;
  //b->radius = 2.0 + sqrt(sqrt(speed));
  sum = 0;
  b->radius = 7*noise + 4.0 + b->pressure / 2.0;
  b->radius = b->radius * b->pressure + 0.01;
#if 0
  i = 0;
  b->opaque  = 0;
  b->opaque += b->weights[i++] * b->pressure;
  b->opaque += b->weights[i++] * speed;
  b->opaque += b->weights[i++] * 1.0;
  b->opaque += b->weights[i++] * noise;
  b->radius  = 0;
  b->radius += b->weights[i++] * b->pressure;
  b->radius += b->weights[i++] * speed;
  b->radius += b->weights[i++] * 1.0;
  b->radius += b->weights[i++] * noise;
  g_assert (i == F_WEIGHTS);
#endif

  if (b->radius > 200) b->radius = 200;
  if (b->radius < 0) b->radius = 0;
  if (b->opaque < 0) b->opaque = 0;
  if (b->opaque > 1) b->opaque = 1;
}

float brush_count_dabs_to (Brush * b, float x, float y, float pressure, float time)
{
  float dx, dy;
  float d_dist;
  if (b->radius  < 1.2) b->radius = 1.2;
  if (b->spacing < 0.01) b->spacing = 0.01;
  dx = x - b->x;
  dy = y - b->y;
  // FIXME: could count time a bit, too...
  d_dist = sqrt (dx*dx + dy*dy);
  return d_dist / (b->spacing*b->radius);
}

void brush_stroke_to (Brush * b, Surface * s, float x, float y, float pressure, float time)
{
  if (time <= b->time) return;

  if (b->time == 0) {
    // reset
    b->x = x;
    b->y = y;
    b->pressure = pressure;
    b->time = time;
    brush_dab (b, s);
    return;
  }

  // draw many (or zero) dabs to the next position
  b->dist += brush_count_dabs_to (b, x, y, pressure, time);
  while (b->dist >= 1.0) {
    float step; // percent of distance between current and next position
    step = 1 / b->dist;
    // linear interpolation
    b->dx        = step * (x - b->x);
    b->dy        = step * (y - b->y);
    b->dpressure = step * (pressure - b->pressure);
    b->dtime     = step * (time - b->time);
    b->x        += b->dx;
    b->y        += b->dy;
    b->pressure += b->dpressure;
    b->time     += b->dtime;

    b->dist     -= 1.0;
    
    brush_prepare_dab (b);
    brush_dab (b, s);
  }
}

