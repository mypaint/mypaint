#ifndef __brush_h__
#define __brush_h__

#include "surface.h"

typedef struct {
  // lowlevevel stuff
  float x, y, pressure, time;
  float dx, dy, dpressure, dtime;
  float dist;
  float tmp_one_over_radius;
  
  GtkWidget * queue_draw_widget;

  // user settings
  float radius_logarithmic; // log(radius)
  float opaque; // 1 - transparency
  guchar color[3];

  float dabs_per_radius; // 1/spacing
  float dabs_per_second;
} Brush;

// brush setting flags
#define BSF_NONE 0
#define BSF_LOGARITHMIC 1

typedef struct {
  char * cname;
  char * name;
  int flags;
  float min;
  float default_value; // "default" seems to be a keyword...
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
