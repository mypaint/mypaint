#ifndef __brush_h__
#define __brush_h__

#include "surface.h"

// brush setting flags (not yet used)
#define BSF_NONE 0
#define BSF_LOGARITHMIC 1

// only accessed inside brush.c
typedef struct {
  // lowlevevel stuff
  float x, y, pressure, time;
  float dx, dy, dpressure, dtime;
  float dist;
  float actual_radius;
  
  GtkWidget * queue_draw_widget;

  guchar color[3];

  // User settings, input for generate.py:
  // name % flags % min % default % max % tooltip
  //% opaque % BSF_NONE % 0.0 % 1.0 % 1.0 % 0 means brush is transparent, 1 fully visible
  float opaque;
  //% radius % BSF_LOGARITHMIC % -0.5 % 2.0 % 5.0 % basic brush radius
  float radius_logarithmic;
  //% dabs per basic radius % BSF_NONE % 0.0 % 0.0 % 30.0 % dabs to draw while the pointer moves one brush radius
  float dabs_per_basic_radius;
  //% dabs per actual radius % BSF_NONE % 0.0 % 2.0 % 30.0 % same as above, but the radius actually drawn is used, which might change dynamically
  float dabs_per_actual_radius;
  //% dabs per second % BSF_NONE % 0.0 % 0.0 % 80.0 % dabs to draw each second, no matter how far the pointer moves
  float dabs_per_second;
} Brush;

typedef struct {
  char * cname;
  char * name;
  int flags;
  float min;
  float default_value;
  float max;
  char * helptext;
} BrushSettingInfo;

extern BrushSettingInfo brush_setting_infos[];

extern Brush * brush_create ();
extern Brush * brush_create_copy (Brush * b);
extern void brush_free (Brush * b);
extern void brush_reset (Brush * b);
extern void brush_stroke_to (Brush * b, Surface * s, float x, float y, float pressure, float time);

extern void brush_set_setting (Brush * b, int id, float value);
extern float brush_get_setting (Brush * b, int id);

extern void brush_mutate (Brush * b);

#endif
