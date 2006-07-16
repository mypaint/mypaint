#ifndef __GTK_MY_DRAW_WIDGET_H__
#define __GTK_MY_DRAW_WIDGET_H__

#include <gdk/gdk.h>
#include <gtk/gtkwidget.h>
#include <gtk/gtkdrawingarea.h>
#include <gdk-pixbuf/gdk-pixbuf.h>

#include "surface.h"
#include "gtkmybrush.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */


#define GTK_TYPE_MY_DRAW_WIDGET            (gtk_my_draw_widget_get_type ())
#define GTK_MY_DRAW_WIDGET(obj)            (G_TYPE_CHECK_INSTANCE_CAST ((obj), GTK_TYPE_MY_DRAW_WIDGET, GtkMyDrawWidget))
#define GTK_MY_DRAW_WIDGET_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST ((klass), GTK_TYPE_MY_DRAW_WIDGET, GtkMyDrawWidgetClass))
#define GTK_IS_MY_DRAW_WIDGET(obj)         (G_TYPE_CHECK_INSTANCE_TYPE ((obj), GTK_TYPE_MY_DRAW_WIDGET))
#define GTK_IS_MY_DRAW_WIDGET_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE ((klass), GTK_TYPE_MY_DRAW_WIDGET))
#define GTK_MY_DRAW_WIDGET_GET_CLASS(obj)  (G_TYPE_INSTANCE_GET_CLASS ((obj), GTK_TYPE_MY_DRAW_WIDGET, GtkMyDrawWidgetClass))


typedef struct _GtkMyDrawWidget       GtkMyDrawWidget;
typedef struct _GtkMyDrawWidgetClass  GtkMyDrawWidgetClass;

struct _GtkMyDrawWidget
{
  GtkDrawingArea widget;

  Surface * surface;
  GtkMyBrush * brush;

  float viewport_x, viewport_y;
  float zoom, one_over_zoom;

  int allow_dragging;
  int dragging;
  float dragging_last_x, dragging_last_y;
};

struct _GtkMyDrawWidgetClass
{
  GtkDrawingAreaClass parent_class;

  void (*dragging_finished) (GtkMyDrawWidget *mdw);
};


GType      gtk_my_draw_widget_get_type   (void) G_GNUC_CONST;

GtkWidget* gtk_my_draw_widget_new        (void);

void gtk_my_draw_widget_clear (GtkMyDrawWidget *mdw);
void gtk_my_draw_widget_set_brush (GtkMyDrawWidget *mdw, GtkMyBrush * brush);
void gtk_my_draw_widget_set_viewport (GtkMyDrawWidget *mdw, float x, float y);
float gtk_my_draw_widget_get_viewport_x (GtkMyDrawWidget *mdw);
float gtk_my_draw_widget_get_viewport_y (GtkMyDrawWidget *mdw);
void gtk_my_draw_widget_allow_dragging (GtkMyDrawWidget *mdw, int allow);
void gtk_my_draw_widget_set_zoom (GtkMyDrawWidget *mdw, float zoom);
float gtk_my_draw_widget_get_zoom (GtkMyDrawWidget *mdw);

GdkPixbuf* gtk_my_draw_widget_get_as_pixbuf (GtkMyDrawWidget *mdw);
GdkPixbuf* gtk_my_draw_widget_get_nonwhite_as_pixbuf (GtkMyDrawWidget *mdw);
void gtk_my_draw_widget_set_from_pixbuf (GtkMyDrawWidget *mdw, GdkPixbuf* pixbuf);
void gtk_my_draw_widget_discard_and_resize (GtkMyDrawWidget *mdw, int width, int height);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* __GTK_MY_DRAW_WIDGET_H__ */
