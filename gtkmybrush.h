#ifndef __GTK_MY_BRUSH_H__
#define __GTK_MY_BRUSH_H__

#include <glib.h>
#include <glib-object.h>
#include <gdk/gdk.h>
#include <gtk/gtkwidget.h>

#include "surface.h"
#include "brushsettings.h"

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

typedef struct {
  // a set of control points (stepwise linear)
  float xvalues[4]; // > 0 because all inputs are > 0
  float yvalues[4]; // range: -oo  .. +oo (added to base_value)
  // xvalues can be zero to indicate that this point is not used.
  // the first point (0, 0) is implicit, would have index -1
} Mapping;
  
typedef struct {
  float base_value;
  // NULL if it does not depend on this input
  Mapping * mapping[INPUT_COUNT];
} Setting;

struct _GtkMyBrush
{
  GObject parent;

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
  Setting settings[BRUSH_SETTINGS_COUNT];
  // the resulting values
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
void gtk_my_brush_remove_mapping (GtkMyBrush * b, int id, int input);
void gtk_my_brush_set_color (GtkMyBrush * b, int red, int green, int blue);
void gtk_my_brush_set_print_inputs (GtkMyBrush * b, int value);
float gtk_my_brush_get_painting_time (GtkMyBrush * b);
void gtk_my_brush_set_painting_time (GtkMyBrush * b, float value);

GdkPixbuf* gtk_my_brush_get_colorselection_pixbuf (GtkMyBrush * b);

/* only for mydrawwidget (not exported to python): */
void brush_stroke_to (GtkMyBrush * b, Surface * s, float x, float y, float pressure, double time, Rect * bbox);
void brush_reset (GtkMyBrush * b);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* __GTK_MY_BRUSH_H__ */
