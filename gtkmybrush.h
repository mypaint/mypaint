#ifndef __GTK_MY_BRUSH_H__
#define __GTK_MY_BRUSH_H__

#include <glib.h>
#include <glib-object.h>
#include <gdk/gdk.h>
#include <gtk/gtkwidget.h>

#include "surface.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */


#define GTK_TYPE_MY_BRUSH            (gtk_my_brush_get_type ())
#define GTK_MY_BRUSH(obj)            (G_TYPE_CHECK_INSTANCE_CAST ((obj), GTK_TYPE_MY_BRUSH, GtkMyBrush))
#define GTK_MY_BRUSH_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST ((klass), GTK_TYPE_MY_BRUSH, GtkMyBrushClass))
#define GTK_IS_MY_BRUSH(obj)         (G_TYPE_CHECK_INSTANCE_TYPE ((obj), GTK_TYPE_MY_BRUSH))
#define GTK_IS_MY_BRUSH_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE ((klass), GTK_TYPE_MY_BRUSH))
#define GTK_MY_BRUSH_GET_CLASS(obj)  (G_TYPE_INSTANCE_GET_CLASS ((obj), GTK_TYPE_MY_BRUSH, GtkMyBrushClass))


typedef struct _GtkMyBrush       GtkMyBrush;
typedef struct _GtkMyBrushClass  GtkMyBrushClass;

struct _GtkMyBrush
{
  GObject parent;

  // lowlevevel stuff
  float x, y, pressure, time;
  float dx, dy, dpressure, dtime;
  float dist;
  float actual_radius;

  GtkWidget * queue_draw_widget;

  guchar color[3];

  // misc helpers
  float x_slow, y_slow;
  float dx_slow, dy_slow;
  float last_time;
  float obs__speedabs_slow;
  float rbs__speedabs_slow;

  // User settings, input for generate.py:
  // name % flags % min % default % max % tooltip
  //% opaque % BSF_NONE % 0.0 % 1.0 % 1.0 % 0 means brush is transparent, 1 fully visible
  float opaque;
  //% radius % BSF_LOGARITHMIC % -0.5 % 2.0 % 5.0 % basic brush radius (logarithmic)\n 0.7 means 2 pixels\n 3.0 means 20 pixels
  float radius_logarithmic;
  //% hardness % BSF_NONE % 0.0 % 1.0 % 1.0 % hard brush-circle borders (setting to zero will draw nothing; it's not implemented like in GIMP, I haven't figured out yet)
  float hardness;
  //% dabs per basic radius % BSF_NONE % 0.0 % 0.0 % 5.0 % dabs to draw while the pointer moves one brush radius
  float dabs_per_basic_radius;
  //% dabs per actual radius % BSF_NONE % 0.0 % 2.0 % 5.0 % same as above, but the radius actually drawn is used, which might change dynamically
  float dabs_per_actual_radius;
  //% dabs per second % BSF_NONE % 0.0 % 0.0 % 80.0 % dabs to draw each second, no matter how far the pointer moves
  float dabs_per_second;
  //% opaque by pressure % BSF_NONE % 0.0 % 1.0 % 5.0 % opaque above will get multiplyed by pressure times this value\nFIXME: this is really useless and has the same effect as oppaque.
  float opaque_by_pressure;
  //% radius by pressure % BSF_NONE % -10.0 % 0.1 % 10.0 % how much more pressure will increase the radius\nwithout pressure, the radius is unchanged\n 0.0 disable\n 0.7 double radius at full pressure\n-0.7 half radius at full pressure\n3.0 20 times radius at full pressure
  float radius_by_pressure;
  //% radius by random % BSF_NONE % 0.0 % 0.0 % 10.0 % alter the radius randomly each dab\n 0.0 disable\n 0.7 biggest radius is twice as large as smallest\n 3.0 biggest radius 20 times as large as smallest
  float radius_by_random;
  //% radius by speed % BSF_NONE % -10.0 % 0.0 % 10.0 % alter the radius depending on current speed; this is also affected by 'speed abs slowness' below, but not by 'speed slowness'
  float radius_by_speed;
  //% radius by speed: speed abs slowness % BSF_NONE % 0.0 % 0.0 % 10.0 % how slow to update the speed value\n0.0 change the radius immediatly as your speed changes
  float rbs__speedabs_slowness;
  //% offset by random % BSF_NONE % 0.0 % 0.0 % 10.0 % add randomness to the position where the dab is drawn\n 0.0 disabled\n 1.0 standard derivation is one radius away (as set above, not the actual radius)
  float offset_by_random;
  //% offset by speed % BSF_NONE % -30.0 % 0.0 % 30.0 % change position depending on pointer speed\n= 0 disable\n> 0 draw where the pointer moves to\n< 0 draw where the pointer comes from
  float offset_by_speed;
  //% offset by speed: speed slowness % BSF_NONE % 0.0 % 0.0 % 10.0 % use a short-term speed (0) or a long time average speed (big) for above
  float obs__speed_slowness;
  //% offset by speed: speed abs slowness % BSF_NONE % 0.0 % 0.0 % 10.0 % how fast to adapt the absolut value of the speed (in contrast to the direction)
  float obs__speedabs_slowness;
  //% saturation slowdown % BSF_NONE % -1.0 % 0.0 % 1.0 % When painting black, it soon gets black completely. This setting controls how fast the final brush color is taken:\n 1.0 slowly\n 0.0 disable\n-1.0 even faster\nThis is nolinear and causes strange effects when it happens too fast. Set occupancy low enough to avoid this.\nFor example, a full-occupancy black stroke might get brighter over grey areas than over white ones.\nFIXME: this setting seems not to work as I expected. I reccomend to set it to zero for anything else than black/white drawing.
  float saturation_slowdown;
  //% slow position % BSF_NONE % 0.0 % 0.0 % 10.0 % Slowdown pointer tracking speed. 0 disables it, higher values remove more jitter in cursor movements. Useful for drawing smooth, comic-like outlines.
  float position_T;
  //% slow position 2 % BSF_NONE % 0.0 % 0.0 % 10.0 % Similar as above but at brushdab level (ignoring how much time has past, if brushdabs do not depend on time)
  float position_T2;
  //% color brightness by pressure % BSF_NONE % -2.0 % 0.0 % 2.0 % change the color brightness (also known as intensity or value) depending on pressure\n-1.0 high pressure: darker\n 0.0 disable\n 1.0 high pressure: brigher
  float color_value_by_pressure;
  //% color brightness by random % BSF_NONE % 0.0 % 0.0 % 1.0 % noisify the color brightness (also known as intensity or value)
  float color_value_by_random;
  //% color saturation by pressure % BSF_NONE % -2.0 % 0.0 % 2.0 % change the color saturation depending on pressure\n-1.0 high pressure: grayish\n 0.0 disable\n 1.0 high pressure: saturated
  float color_saturation_by_pressure;
  //% color saturation by random % BSF_NONE % 0.0 % 0.0 % 1.0 % noisify the color saturation
  float color_saturation_by_random;
  //% color hue by pressure % BSF_NONE % -2.0 % 0.0 % 2.0 % change color hue depending on pressure\n-1.0 high pressure: clockwise color hue shift\n 0.0 disable\n 1.0 high pressure: counterclockwise hue shift
  float color_hue_by_pressure;
  //% color hue by random % BSF_NONE % 0.0 % 0.0 % 1.0 % noisify the color hue
  float color_hue_by_random;
  //% adapt color from image % BSF_NONE % 0.0 % 0.0 % 1.0 % slowly change the color to the one you're painting on (some kind of smudge tool)\nNote that this happens /before/ the hue/saturation/brighness adjustment below: you can get very different effects (eg brighten image) by combining with them.
  float adapt_color_from_image;
  //TODO second dab % BSF_NONE % 0.0 % 0.0 % 1.0 % whether to draw a second brusdab next to the normal one, see options below
  float second_dab;
  //TODO second dab angle % BSF_NONE % 0.0 % 180.0 % 360.0 % angle in degree where the second dab is placed
  float second_dab_angle;
  //TODO second dab offset % BSF_NONE % -5.0 % 1.0 % 5.0 % how far away (multiplied by actual brush radius) the second dab is from the first dab
  float second_dab_offset;
  //TODO: second dab value adjustment..., change angle by speed angle...
};

struct _GtkMyBrushClass
{
  GObjectClass parent_class;
};


GType       gtk_my_brush_get_type   (void) G_GNUC_CONST;
GtkMyBrush* gtk_my_brush_new        (void);

void gtk_my_brush_set_setting (GtkMyBrush * b, int id, float value);
float gtk_my_brush_get_setting (GtkMyBrush * b, int id);

void gtk_my_brush_set_color (GtkMyBrush * b, int red, int green, int blue);

// only for mydrawwidget (not exported to python):
void brush_stroke_to (GtkMyBrush * b, Surface * s, float x, float y, float pressure, float time);
void brush_reset (GtkMyBrush * b);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* __GTK_MY_BRUSH_H__ */
