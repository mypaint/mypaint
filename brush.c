#include <stdio.h>
#include <string.h>
#include <glib.h>
#include <math.h>
#include "brush.h"

BrushSettingInfo brush_setting_infos[] = {
  // cname, name, flags, min, default_value, max, helptext
  { "radius_logarithmic", "Radius", BSF_LOGARITHMIC, -0.5, 2.0, 5.0, "basic brush radius" },
  { "opaque", "Opaque", BSF_NONE, 0.0, 1.0, 1.0, "0 means brush is transparent, 1 fully visible" },
  { "dabs_per_radius", "Dabs per Radius", BSF_NONE, 0.0, 2.0, 50.0, "number of brushdabs to do while the mouse moves a distance of one radius (1/spacing)" },
  { "dabs_per_second", "Dabs per Second", BSF_NONE, 0.0, 0.0, 50.0, "number of brushdabs to each second, no matter whether the mouse moved" },
  { 0, 0, 0, 0, 0, 0, 0 }
};

#include "brush_settings.inc"

Brush * brush_create ()
{
  int i;
  Brush * b;
  b = g_new0 (Brush, 1);
  b->queue_draw_widget = NULL;
  for (i=0; brush_setting_infos[i].cname; i++) {
    brush_set_setting (b, i, brush_setting_infos[i].default_value);
  }
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
#if 0
  int i;
  for (i=0; i<F_WEIGHTS; i++) {
    /*
    if (g_random_int_range(0, 10) == 0) {
      b->variations[i] *= g_random_double_range(0.5, 1.0/0.5);
      }*/
    b->weights[i] += g_random_double_range(-b->variations[i], b->variations[i]);
  }
#endif
}

void draw_brush_dab (Surface * s, GtkWidget * queue_draw_widget, 
                     float x, float y, 
                     float radius, float opaque_float,
                     guchar * color) {
  float r_fringe;
  int x0, y0;
  int x1, y1;
  int xp, yp;
  guchar *rgb;
  float xx, yy, rr;
  float radius2;
  int opaque;
  guchar c[3];
  if (!s) return;

  r_fringe = radius + 1;
  x0 = floor (x - r_fringe);
  y0 = floor (y - r_fringe);
  x1 = ceil (x + r_fringe);
  y1 = ceil (y + r_fringe);
  if (x0 < 0) x0 = 0;
  if (y0 < 0) y0 = 0;
  if (x1 > s->w) x1 = s->w;
  if (y1 > s->h) y1 = s->h;
  rr = sqr(radius);
  opaque = floor(opaque_float * 256 + 0.5);
  if (opaque <= 0) return;
  if (opaque > 256) opaque = 256;
  c[0] = color[0];
  c[1] = color[1];
  c[2] = color[2];
  radius2 = sqr(radius);

  for (yp = y0; yp < y1; yp++) {
    yy = (yp + 0.5 - y);
    yy *= yy;
    for (xp = x0; xp < x1; xp++) {
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
  
  if (queue_draw_widget) {
    gtk_widget_queue_draw_area (queue_draw_widget,
                                floor(x - (radius+1)),
                                floor(y - (radius+1)),
                                /* FIXME: think about it exactly */
                                ceil (2*(radius+1)),
                                ceil (2*(radius+1))
                                );
  }
}

// high-level part of before each dab
void brush_prepare_and_draw_dab (Brush * b, Surface * s)
{
  float x, y, radius_log, radius, opaque;
  float speed;
  float noise;

  x = b->x; y = b->y;
  radius_log = b->radius_logarithmic;
  opaque = b->opaque;
  
  noise = g_random_double () - 0.5; // [-0.5..0.5)
  speed = sqrt(sqr(b->dx) + sqr(b->dy))/b->dtime;
  opaque *= b->pressure / 8.0; // TODO: make configurable
  //b->radius = 2.0 + sqrt(sqrt(speed));
  radius_log += 0.1 * b->pressure; // TODO: make configurable
  radius_log += noise; // TODO: make configurable

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

  radius = exp(radius_log);

  g_assert(radius > 0);
  if (radius > 200) radius = 200;
  g_assert(opaque >= 0);
  g_assert(opaque <= 1);

  // used for interpolation later
  b->tmp_one_over_radius = 1/radius;
  
  draw_brush_dab (s, b->queue_draw_widget,
                  x, y, radius, opaque, b->color);
}

float brush_count_dabs_to (Brush * b, float x, float y, float pressure, float time)
{
  float dx, dy, dt;
  float res1, res2;
  if (b->tmp_one_over_radius == 0) b->tmp_one_over_radius = 1.0/exp(b->radius_logarithmic);
  if (b->tmp_one_over_radius > 2) b->tmp_one_over_radius = 2;
  if (b->tmp_one_over_radius < 1/500.0) b->tmp_one_over_radius = 1/500.0;
  dx = x - b->x;
  dy = y - b->y;
  //dp = pressure - b->pressure; // Not useful?
  dt = time - b->time;

  res1 = sqrt (dx*dx + dy*dy) * b->tmp_one_over_radius * b->dabs_per_radius;
  res2 = dt * b->dabs_per_second;
  return res1 + res2;
}

void brush_stroke_to (Brush * b, Surface * s, float x, float y, float pressure, float time)
{
  if (time <= b->time) return;

  if (b->time == 0 || time - b->time > 10) {
    // reset
    b->x = x;
    b->y = y;
    b->pressure = pressure;
    b->time = time;
    //brush_dab (b, s);
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
    
    brush_prepare_and_draw_dab (b, s);
  }
}

