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
  Surface * surface;
} Brush;

extern Brush * brush_create (Surface * surface);
extern void brush_free (Brush * b);
extern void brush_stroke_to (Brush * b, float x, float y, float pressure, float time);

#endif
