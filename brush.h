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
  //% radius % BSF_LOGARITHMIC % -0.5 % 2.0 % 5.0 % basic brush radius (logarithmic)\n 0.7 means 2 pixels\n 3.0 means 20 pixels
  float radius_logarithmic;
  //% hardness % BSF_NONE % 0.0 % 1.0 % 1.0 % 0 hard brush-circle borders, 1 fuzzy borders
  float hardness;
  //% dabs per basic radius % BSF_NONE % 0.0 % 0.0 % 5.0 % dabs to draw while the pointer moves one brush radius
  float dabs_per_basic_radius;
  //% dabs per actual radius % BSF_NONE % 0.0 % 2.0 % 5.0 % same as above, but the radius actually drawn is used, which might change dynamically
  float dabs_per_actual_radius;
  //% dabs per second % BSF_NONE % 0.0 % 0.0 % 80.0 % dabs to draw each second, no matter how far the pointer moves
  float dabs_per_second;
  //% opaque by pressure % BSF_NONE % 0.0 % 1.0 % 1.0 % 0.0 means opaque stays as given above\n1.0 means opaque is multiplied by pressure
  float opaque_by_pressure;
  //% radius by pressure % BSF_NONE % -10.0 % 0.1 % 10.0 % how much more pressure will increase the radius\nwithout pressure, the radius is unchanged\n 0.0 disable\n 0.7 double radius at full pressure\n-0.7 half radius at full pressure\n3.0 20 times radius at full pressure
  float radius_by_pressure;
  //% radius by random % BSF_NONE % 0.0 % 0.0 % 10.0 % alter the radius randomly each dab\n 0.0 disable\n 0.7 biggest radius is twice as large as smallest\n 3.0 biggest radius 20 times as large as smallest
  float radius_by_random;
  //% offset by random % BSF_NONE % 0.0 % 0.0 % 10.0 % add randomness to the position where the dab is drawn\n 0.0 disabled\n 1.0 standard derivation is one radius away (as set above, not the actual radius)
  float offset_by_random;
  //% offset by speed % BSF_NONE % -30.0 % 0.0 % 30.0 % change position depending on pointer speed\n= 0 disable\n> 0 draw where the pointer moves to\n< 0 draw where the pointer comes from
  float offset_by_speed;
  //% saturation slowdown % BSF_NONE % -1.0 % 0.0 % 1.0 % when painting black, it soon gets black completely; this setting controls how fast the final brush color is taken\n 1.0 slowly\n 0.0 disable\n-1.0 even faster
  float saturation_slowdown;
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
