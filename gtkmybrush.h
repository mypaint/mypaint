#ifndef __GTK_MY_BRUSH_H__
#define __GTK_MY_BRUSH_H__

#include <glib.h>
#include <glib-object.h>
#include <gdk/gdk.h>
#include <gtk/gtkwidget.h>

#include "gtkmysurfaceold.h"
#include "brushsettings.h"
#include "mapping.h"


#define GTK_TYPE_MY_BRUSH            (gtk_my_brush_get_type ())
#define GTK_MY_BRUSH(obj)            (G_TYPE_CHECK_INSTANCE_CAST ((obj), GTK_TYPE_MY_BRUSH, GtkMyBrush))
#define GTK_MY_BRUSH_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST ((klass), GTK_TYPE_MY_BRUSH, GtkMyBrushClass))
#define GTK_IS_MY_BRUSH(obj)         (G_TYPE_CHECK_INSTANCE_TYPE ((obj), GTK_TYPE_MY_BRUSH))
#define GTK_IS_MY_BRUSH_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE ((klass), GTK_TYPE_MY_BRUSH))
#define GTK_MY_BRUSH_GET_CLASS(obj)  (G_TYPE_INSTANCE_GET_CLASS ((obj), GTK_TYPE_MY_BRUSH, GtkMyBrushClass))


typedef struct _GtkMyBrush       GtkMyBrush;
typedef struct _GtkMyBrushClass  GtkMyBrushClass;

/* The GtkMyBrush structure (gobject) stores two things:
   a) the states of the cursor (velocity, color, speed)
   b) the brush settings (as set in the GUI)
   FIXME: Actually those are two orthogonal things. Should separate them.

   In python, there are two kinds of instances from this: a "global
   brush" which does the cursor tracking, and the "brushlist" where
   only the settings are important. When a brush is selected, its
   settings are copied into the global one, leaving the status intact.
 */

struct _GtkMyBrush
{
  GObject parent;

  GtkMySurface * target_surface;

  // lowlevevel stuff (almost raw input)
  float x, y, pressure; double time;
  float dx, dy, dpressure, dtime; // note: this is dx/ddab, ..., dtime/ddab (dab number, 5.0 = 5th dab)
  float dist;
  float base_radius;
  float actual_radius;
  double last_time;

  guchar color[3];

  int print_inputs;

  // misc higher-level helper variables
  float actual_x, actual_y; // for slow position
  float norm_dx_slow, norm_dy_slow; // note: now this is dx/dt * (1/radius)

  float norm_speed_slow1; 
  float norm_speed_slow2;

  float stroke;
  int stroke_started;

  float custom_input;

  float painting_time;

  // description how to calculate the values
  Mapping * settings[BRUSH_SETTINGS_COUNT];

  // the current value of a setting
  // FIXME: they could as well be passed as parameters to the dab function
  //        (Hm. This way no malloc is needed before each dab. Think about that.)
  float settings_value[BRUSH_SETTINGS_COUNT];

};

struct _GtkMyBrushClass
{
  GObjectClass parent_class;
};


GType       gtk_my_brush_get_type   (void) G_GNUC_CONST;
GtkMyBrush* gtk_my_brush_new        (void);

/* no getter functions since values are remembered in python code */
void gtk_my_brush_set_base_value (GtkMyBrush * b, int id, float value);
void gtk_my_brush_set_mapping (GtkMyBrush * b, int id, int input, int index, float value);
void gtk_my_brush_set_color (GtkMyBrush * b, int red, int green, int blue);
void gtk_my_brush_set_print_inputs (GtkMyBrush * b, int value);
float gtk_my_brush_get_painting_time (GtkMyBrush * b);
void gtk_my_brush_set_painting_time (GtkMyBrush * b, float value);

GdkPixbuf* gtk_my_brush_get_colorselection_pixbuf (GtkMyBrush * b);

/* only for mydrawwidget (not exported to python): */
void brush_stroke_to (GtkMyBrush * b, GtkMySurfaceOld * s, float x, float y, float pressure, double dtime);
void brush_reset (GtkMyBrush * b);

/* note: currently the random seed is global, not brush specific */
void gtk_my_brush_srandom (GtkMyBrush * b, int value);
double gtk_my_brush_random_double (GtkMyBrush * b); // for testing the RNG

#endif /* __GTK_MY_BRUSH_H__ */
