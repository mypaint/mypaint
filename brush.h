#ifndef __brush_h__
#define __brush_h__

#include "surface.h"

typedef struct {
  float x, y, pressure, time;
  float dx, dy, dpressure, dtime;
  float radius;
  float opaque;
  guchar color[3];
  float spacing;
  float dist;

#define F_INPUTS  4 /* pressure, speed, noise, bias */
#define F_OUTPUTS 2 /* opaque, size */
#define F_WEIGHTS (F_INPUTS * F_OUTPUTS)
  float weights[F_WEIGHTS];
  float variations[F_WEIGHTS];

  GtkWidget * queue_draw_widget;
} Brush;

extern Brush * brush_create ();
extern Brush * brush_create_copy (Brush * b);
extern void brush_free (Brush * b);
extern void brush_reset (Brush * b);
extern void brush_stroke_to (Brush * b, Surface * s, float x, float y, float pressure, float time);

extern void brush_mutate (Brush * b);

#endif
