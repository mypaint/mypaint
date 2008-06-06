/* This file is part of MyPaint.
 * Copyright (C) 2007 by Martin Renold <martinxyz@gmx.ch>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License.
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY. See the COPYING file for more details.
 */

// gtk stock code - left gtk prefix to use the pygtk wrapper-generator easier
#include "gtkmydrawwidget.h"
#include "stroke_recorder.h"
#include "mapping.h"

static void gtk_my_draw_widget_class_init    (GtkMyDrawWidgetClass *klass);
static void gtk_my_draw_widget_init          (GtkMyDrawWidget      *mdw);
static void gtk_my_draw_widget_finalize (GObject *object);
static void gtk_my_draw_widget_realize (GtkWidget *widget);
static gint gtk_my_draw_widget_button_updown (GtkWidget *widget, GdkEventButton *event);
static gint gtk_my_draw_widget_motion_notify (GtkWidget *widget, GdkEventMotion *event);
static gint gtk_my_draw_widget_proximity_inout (GtkWidget *widget, GdkEventProximity *event);
static gint gtk_my_draw_widget_expose (GtkWidget *widget, GdkEventExpose *event);
static void gtk_my_draw_widget_surface_modified (GtkMySurface *s, gint x, gint y, gint w, gint h, GtkMyDrawWidget *mdw);


static gpointer parent_class;

enum {
  DRAGGING_FINISHED,
  LAST_SIGNAL
};
guint gtk_my_draw_widget_signals[LAST_SIGNAL] = { 0 };

GType
gtk_my_draw_widget_get_type (void)
{
  static GType type = 0;

  if (!type)
    {
      static const GTypeInfo info =
      {
	sizeof (GtkMyDrawWidgetClass),
	NULL,		/* base_init */
	NULL,		/* base_finalize */
	(GClassInitFunc) gtk_my_draw_widget_class_init,
	NULL,		/* class_finalize */
	NULL,		/* class_data */
	sizeof (GtkMyDrawWidget),
	0,		/* n_preallocs */
	(GInstanceInitFunc) gtk_my_draw_widget_init,
      };

      type =
	g_type_register_static (GTK_TYPE_DRAWING_AREA, "GtkMyDrawWidget",
				&info, 0);
    }

  return type;
}

static void
gtk_my_draw_widget_class_init (GtkMyDrawWidgetClass *class)
{
  GtkWidgetClass *widget_class = GTK_WIDGET_CLASS (class);
  GObjectClass *gobject_class = G_OBJECT_CLASS (class);

  parent_class = g_type_class_peek_parent (class);
        
  gobject_class->finalize = gtk_my_draw_widget_finalize;
  widget_class->realize = gtk_my_draw_widget_realize;

  widget_class->expose_event = gtk_my_draw_widget_expose;
  widget_class->motion_notify_event = gtk_my_draw_widget_motion_notify;
  widget_class->button_press_event = gtk_my_draw_widget_button_updown;
  widget_class->button_release_event = gtk_my_draw_widget_button_updown;
  widget_class->proximity_in_event = gtk_my_draw_widget_proximity_inout;
  widget_class->proximity_out_event = gtk_my_draw_widget_proximity_inout;


  gtk_my_draw_widget_signals[DRAGGING_FINISHED] = g_signal_new 
    ("dragging-finished",
     G_TYPE_FROM_CLASS (class),
     G_SIGNAL_RUN_LAST,
     G_STRUCT_OFFSET (GtkMyDrawWidgetClass, dragging_finished),
     NULL, NULL,
     g_cclosure_marshal_VOID__VOID,
     G_TYPE_NONE, 0);

}

static void
gtk_my_draw_widget_realize (GtkWidget *widget)
{
  GtkMyDrawWidget *mdw;
  GdkWindowAttr attributes;
  gint attributes_mask;

  g_return_if_fail (GTK_IS_MY_DRAW_WIDGET (widget));

  mdw = GTK_MY_DRAW_WIDGET (widget);
  GTK_WIDGET_SET_FLAGS (widget, GTK_REALIZED);

  attributes.window_type = GDK_WINDOW_CHILD;
  attributes.x = widget->allocation.x;
  attributes.y = widget->allocation.y;
  attributes.width = widget->allocation.width;
  attributes.height = widget->allocation.height;
  attributes.wclass = GDK_INPUT_OUTPUT;
  attributes.visual = gtk_widget_get_visual (widget);
  attributes.colormap = gtk_widget_get_colormap (widget);

  attributes.event_mask = gtk_widget_get_events (widget);
  attributes.event_mask |= (GDK_EXPOSURE_MASK |
                            GDK_LEAVE_NOTIFY_MASK |
                            GDK_BUTTON_PRESS_MASK |
                            GDK_BUTTON_RELEASE_MASK |
                            GDK_POINTER_MOTION_MASK |
                            GDK_PROXIMITY_IN_MASK |
                            GDK_PROXIMITY_OUT_MASK);

  attributes_mask = GDK_WA_X | GDK_WA_Y | GDK_WA_VISUAL | GDK_WA_COLORMAP;

  widget->window = gdk_window_new (gtk_widget_get_parent_window (widget), &attributes, attributes_mask);
  gdk_window_set_user_data (widget->window, mdw);

  widget->style = gtk_style_attach (widget->style, widget->window);
  gtk_style_set_background (widget->style, widget->window, GTK_STATE_NORMAL);

  // needed for some unknown reason
  gtk_widget_add_events (widget, attributes.event_mask);
  // needed for known reason
  gtk_widget_set_extension_events (widget, GDK_EXTENSION_EVENTS_ALL);

  //gtk_drawing_area_send_configure (GTK_DRAWING_AREA (widget));
}

static void
gtk_my_draw_widget_finalize (GObject *object)
{
  GtkMyDrawWidget * mdw;
  g_return_if_fail (object != NULL);
  g_return_if_fail (GTK_IS_MY_DRAW_WIDGET (object));
  mdw = GTK_MY_DRAW_WIDGET (object);
  // can be called multiple times
  if (mdw->surface) {
    g_signal_handlers_disconnect_by_func (mdw->surface, gtk_my_draw_widget_surface_modified, mdw);
    g_object_unref (mdw->surface);
    mdw->surface = NULL;
  }
  if (mdw->replaying) {
    g_array_free (mdw->replaying, TRUE);
    mdw->replaying = NULL;
  }
  if (mdw->recording) {
    g_array_free (mdw->recording, TRUE);
    mdw->recording = NULL;
  }
  G_OBJECT_CLASS (parent_class)->finalize (object);
}

GtkWidget*
gtk_my_draw_widget_new (void)
{
  return g_object_new (GTK_TYPE_MY_DRAW_WIDGET, NULL);
}

void gtk_my_draw_widget_init (GtkMyDrawWidget *mdw)
{
  mdw->surface = NULL;
  // Dummy width, must be set later using discard_and_resize. 
  // I decided for an empty constructor because [complicated excuse
  // removed] since http://live.gnome.org/PyGTK/WhatsNew28
  gtk_my_draw_widget_discard_and_resize (mdw, 1, 1);

  mdw->zoom = 1.0;
  mdw->one_over_zoom = 1.0;
  mdw->dragging = 0;

  mdw->recording = NULL;
  mdw->replaying = NULL;
  mdw->last_time = 0;
}

void gtk_my_draw_widget_discard_and_resize (GtkMyDrawWidget *mdw, int width, int height)
{
  if (mdw->surface) {
    g_signal_handlers_disconnect_by_func (mdw->surface, gtk_my_draw_widget_surface_modified, mdw);
    g_object_unref (mdw->surface);
  }
  mdw->surface = gtk_my_surface_old_new  (width, height);
  g_signal_connect (mdw->surface, "surface_modified", G_CALLBACK (gtk_my_draw_widget_surface_modified), mdw);

}

Mapping * global_pressure_mapping = NULL;
void global_pressure_mapping_set_n (int n)
{
  if (n == 0) {
    if (global_pressure_mapping) {
      mapping_free(global_pressure_mapping);
      global_pressure_mapping = NULL;
    }
  } else {
    if (!global_pressure_mapping) {
      global_pressure_mapping = mapping_new(1);
    }
    mapping_set_n (global_pressure_mapping, 0, n);
  }
}
void global_pressure_mapping_set_point (int index, float x, float y)
{
  assert(global_pressure_mapping);
  mapping_set_point (global_pressure_mapping, 0, index, x, y);
}

static void
gtk_my_draw_widget_process_motion_or_button (GtkWidget *widget, guint32 time, gdouble x, gdouble y, gdouble pressure)
{
  //RawInput * ri;
  GtkMyDrawWidget * mdw;
  mdw = GTK_MY_DRAW_WIDGET (widget);

  assert(x < 1e8 && y < 1e8 && x > -1e8 && y > -1e8);
  g_assert (pressure >= 0 && pressure <= 1);

  if (global_pressure_mapping) {
    float pressure_input = pressure;
    pressure = mapping_calculate(global_pressure_mapping, &pressure_input);
    g_assert (pressure >= 0 && pressure <= 1);
  }

  if (mdw->dragging) return;

  int dtime;
  if (!mdw->last_time) {
    dtime = 100; //ms
  } else {
    dtime = time - mdw->last_time;
  }
  mdw->last_time = time;

  if (mdw->recording) {
    StrokeEvent e;
    e.dtime = dtime;
    e.x = x;
    e.y = y;
    e.pressure = pressure;
    g_array_append_val (mdw->recording, e);
  }
  
  if (mdw->brush) {
    gtk_my_brush_stroke_to (mdw->brush, mdw->surface,
                            x*mdw->one_over_zoom + mdw->viewport_x, y*mdw->one_over_zoom + mdw->viewport_y,
                            pressure, (double)dtime / 1000.0 /* in seconds */);
  }
}

static void
gtk_my_draw_widget_surface_modified (GtkMySurface *s, gint x, gint y, gint w, gint h, GtkMyDrawWidget *mdw)
{
  x -= (int)(mdw->viewport_x+0.5);
  y -= (int)(mdw->viewport_y+0.5);
  if (mdw->zoom != 1.0) {
    x = (int)(x * mdw->zoom);
    y = (int)(y * mdw->zoom);
    w = (int)(w * mdw->zoom);
    h = (int)(h * mdw->zoom);
    // worst-case rounding problem
    w += 2;
    h += 2;
  }
  gtk_widget_queue_draw_area (GTK_WIDGET (mdw), x, y, w, h);
  //printf ("queued %d %d %d %d\n", x, y, w, h);
}

static gint
gtk_my_draw_widget_button_updown (GtkWidget *widget, GdkEventButton *event)
{
  // button event handling seems to be completely unneccessary
  // (except that you can't make dots with the mouse now)
  return TRUE;
  /*
  GtkMyDrawWidget * mdw;
  g_return_val_if_fail (widget != NULL, FALSE);
  g_return_val_if_fail (GTK_IS_MY_DRAW_WIDGET (widget), FALSE);
  mdw = GTK_MY_DRAW_WIDGET (widget);

  double pressure = 0;
  if (event->type == GDK_BUTTON_PRESS) {
    pressure = 0.5;
  } else if (event->type == GDK_BUTTON_RELEASE) {
    pressure = 0.0;
  } else {
    return FALSE;
  }

  if (event->button != 1) {
    return FALSE;
  }

  double tablet_pressure;
  if (gdk_event_get_axis ((GdkEvent *)event, GDK_AXIS_PRESSURE, &tablet_pressure)) {
    pressure = tablet_pressure;
  }
  printf("but p %f\n", pressure);

  gtk_my_draw_widget_process_motion_or_button (widget, event->time, event->x, event->y, pressure);
  return TRUE;
  */
}

static gint
gtk_my_draw_widget_motion_notify (GtkWidget *widget, GdkEventMotion *event)
{
  GtkMyDrawWidget * mdw;
  g_return_val_if_fail (widget != NULL, FALSE);
  g_return_val_if_fail (GTK_IS_MY_DRAW_WIDGET (widget), FALSE);
  mdw = GTK_MY_DRAW_WIDGET (widget);

  if ((event->state & GDK_BUTTON2_MASK) && mdw->allow_dragging) {
    if (!mdw->dragging) {
      mdw->dragging = 1;
      mdw->dragging_last_x = (int) event->x;
      mdw->dragging_last_y = (int) event->y;
      if (mdw->brush) {
        // cannot record a stroke while dragging, because the brush stores event coordinates
        gtk_my_brush_split_stroke(mdw->brush);
      }
    } else {
      float x, y, dx, dy;
      x = event->x;
      y = event->y;
      dx = x - mdw->dragging_last_x;
      dy = y - mdw->dragging_last_y;
      if (dx == 0 && dy == 0) return TRUE;
      dx *= mdw->one_over_zoom;
      dy *= mdw->one_over_zoom;
      mdw->dragging_last_x = x;
      mdw->dragging_last_y = y;
      gtk_my_draw_widget_set_viewport (mdw, mdw->viewport_x - dx, mdw->viewport_y - dy);
      g_signal_emit (mdw, gtk_my_draw_widget_signals[DRAGGING_FINISHED], 0);
      return TRUE;
    }
  } else {
    if (mdw->dragging) {
      mdw->dragging = 0;
      if (mdw->brush) {
        // cannot record a stroke while dragging, because the brush stores event coordinates
        gtk_my_brush_split_stroke(mdw->brush);
      }
    }
  }

  double pressure;
  if (!gdk_event_get_axis ((GdkEvent *)event, GDK_AXIS_PRESSURE, &pressure)) {
    pressure = (event->state & GDK_BUTTON1_MASK) ? 0.5 : 0;
  }
  gtk_my_draw_widget_process_motion_or_button (widget, event->time, event->x, event->y, pressure);
  return TRUE;
}

static gint
gtk_my_draw_widget_proximity_inout (GtkWidget *widget, GdkEventProximity *event)
{ 
  // handled in python now
  return FALSE;
}

static gint
gtk_my_draw_widget_expose (GtkWidget *widget, GdkEventExpose *event)
{
  GtkMyDrawWidget * mdw;
  guchar *rgb;
  int rowstride;

  g_return_val_if_fail (widget != NULL, FALSE);
  g_return_val_if_fail (GTK_IS_MY_DRAW_WIDGET (widget), FALSE);
  g_return_val_if_fail (event != NULL, FALSE);

  mdw = GTK_MY_DRAW_WIDGET (widget);

  rowstride = event->area.width * 3;
  rowstride = (rowstride + 3) & -4; /* align to 4-byte boundary */
  rgb = g_new (guchar, event->area.height * rowstride);

  //printf("Zoom = %f\n", mdw->zoom);
  if (mdw->zoom == 0.0) mdw->zoom = 1.0; // whyever.
  if (mdw->zoom == 1.0) {
    gtk_my_surface_old_render 
      (mdw->surface,
       rgb, rowstride,
       event->area.x + (int)(mdw->viewport_x+0.5), event->area.y + (int)(mdw->viewport_y+0.5),
       event->area.width, event->area.height,
       /*bpp*/3*8);
  } else {
    gtk_my_surface_old_render_zoom 
      (mdw->surface,
       rgb, rowstride,
       event->area.x + mdw->viewport_x*mdw->zoom, event->area.y + mdw->viewport_y*mdw->zoom,
       event->area.width, event->area.height,
       /*bpp*/3*8,
       mdw->one_over_zoom);
  }

  gdk_draw_rgb_image (widget->window,
		      widget->style->black_gc,
		      event->area.x, event->area.y,
		      event->area.width, event->area.height,
		      GDK_RGB_DITHER_MAX,
		      rgb,
		      rowstride);

  g_free (rgb);
  return FALSE;
}

void	       
gtk_my_draw_widget_clear (GtkMyDrawWidget *mdw)
{
  gtk_my_surface_clear (GTK_MY_SURFACE (mdw->surface));
  gtk_widget_queue_draw (GTK_WIDGET (mdw));
}


GtkMyBrush* 
gtk_my_draw_widget_set_brush (GtkMyDrawWidget *mdw, GtkMyBrush * brush)
{
  GtkMyBrush* brush_old = mdw->brush;
  if (brush) g_object_ref (brush);
  mdw->brush = brush;

  // The caller owns the reference now (caller-owns-return in
  // fix_generated_defs.py) thus we don't g_object_unref here.
  // Cannot return a borrowed reference instead because we just
  // discarded the pointer.
  return brush_old;
}

void gtk_my_draw_widget_allow_dragging (GtkMyDrawWidget *mdw, int allow)
{
  mdw->allow_dragging = allow;
}
void gtk_my_draw_widget_set_viewport (GtkMyDrawWidget *mdw, float x, float y)
{
  mdw->viewport_x = x;
  mdw->viewport_y = y;
  gtk_widget_queue_draw (GTK_WIDGET (mdw));
}
float gtk_my_draw_widget_get_viewport_x (GtkMyDrawWidget *mdw)
{
  return mdw->viewport_x;
}
float gtk_my_draw_widget_get_viewport_y (GtkMyDrawWidget *mdw)
{
  return mdw->viewport_y;
}

void gtk_my_draw_widget_set_zoom (GtkMyDrawWidget *mdw, float zoom)
{
  if (mdw->zoom == zoom) return;
  if (zoom > 0.99 && zoom < 1.01) zoom = 1.0;
  mdw->zoom = zoom;
  mdw->one_over_zoom = 1.0/zoom;
  gtk_widget_queue_draw (GTK_WIDGET (mdw));
}

float gtk_my_draw_widget_get_zoom (GtkMyDrawWidget *mdw)
{
  return mdw->zoom;
}

GdkPixbuf* gtk_my_draw_widget_get_as_pixbuf (GtkMyDrawWidget *mdw)
{
  GdkPixbuf* pixbuf;
  pixbuf = gdk_pixbuf_new (GDK_COLORSPACE_RGB, /*has_alpha*/0, /*bits_per_sample*/8,
			   mdw->surface->w, mdw->surface->h);

  gtk_my_surface_old_render (mdw->surface, 
                             gdk_pixbuf_get_pixels (pixbuf), 
                             gdk_pixbuf_get_rowstride (pixbuf),
                             0, 0, mdw->surface->w, mdw->surface->h,
                             /*bpp*/3*8);

  return pixbuf;
}

GdkPixbuf* gtk_my_draw_widget_get_nonwhite_as_pixbuf (GtkMyDrawWidget *mdw)
{
  Rect r;
  GdkPixbuf* pixbuf;
  gtk_my_surface_old_get_nonwhite_region (mdw->surface, &r);

  pixbuf = gdk_pixbuf_new (GDK_COLORSPACE_RGB, /*has_alpha*/0, /*bits_per_sample*/8,
			   r.w, r.h);

  gtk_my_surface_old_render (mdw->surface, 
                             gdk_pixbuf_get_pixels (pixbuf), 
                             gdk_pixbuf_get_rowstride (pixbuf),
                             r.x, r.y, r.w, r.h,
                             /*bpp*/3*8);

  return pixbuf;
}

void gtk_my_draw_widget_set_from_pixbuf (GtkMyDrawWidget *mdw, GdkPixbuf* pixbuf)
{
  int w, h, n_channels;

  n_channels = gdk_pixbuf_get_n_channels (pixbuf);

  g_assert (gdk_pixbuf_get_colorspace (pixbuf) == GDK_COLORSPACE_RGB);
  g_assert (gdk_pixbuf_get_bits_per_sample (pixbuf) == 8);
  //ignore - g_assert (gdk_pixbuf_get_has_alpha (pixbuf));
  g_assert (n_channels == 4 || n_channels == 3);

  w = gdk_pixbuf_get_width (pixbuf);
  h = gdk_pixbuf_get_height (pixbuf);

  gtk_my_surface_old_load (mdw->surface,
                           gdk_pixbuf_get_pixels (pixbuf),
                           gdk_pixbuf_get_rowstride (pixbuf),
                           w, h,
                           /*bpp*/n_channels*8);
  gtk_widget_queue_draw (GTK_WIDGET (mdw));
}

void gtk_my_draw_widget_start_recording (GtkMyDrawWidget *mdw)
{
  g_assert (!mdw->recording);
  mdw->recording = g_array_new (FALSE, FALSE, sizeof(StrokeEvent));
}

GString* gtk_my_draw_widget_stop_recording (GtkMyDrawWidget *mdw)
{
  // see also mydrawwidget.override
  GString *s;
  s = event_array_to_string (mdw->recording);
  g_array_free (mdw->recording, TRUE); mdw->recording = NULL;
  return s;
}

void gtk_my_draw_widget_stop_replaying (GtkMyDrawWidget *mdw)
{
  g_print ("TODO\n");
  g_assert (!mdw->replaying);
  //mdw->replaying = 
}

void gtk_my_draw_widget_replay (GtkMyDrawWidget *mdw, GString* data, int immediately)
{
  // see also mydrawwidget.override
  if (mdw->replaying) {
    g_print ("Attempting to start replay while replaying.\n");
    return;
  }
  mdw->replaying = string_to_event_array (data);

  if (immediately) {
    int i;
    for (i=0; i<mdw->replaying->len; i++) {
      StrokeEvent *e;
      e = &g_array_index (mdw->replaying, StrokeEvent, i);
      //g_print ("Replay: dtime=%d, x=%f.\n", e->dtime, e->x);
      gtk_my_brush_stroke_to (mdw->brush, mdw->surface,
                              e->x*mdw->one_over_zoom + mdw->viewport_x, e->y*mdw->one_over_zoom + mdw->viewport_y,
                              e->pressure, (double)(e->dtime) / 1000.0 /* in seconds */);
    }
    g_array_free (mdw->replaying, TRUE); mdw->replaying = NULL;
  } else {
    g_print ("TODO\n");
  }
}

/*
GTimer *clock_timer;

...
    clock_timer = g_timer_new();
    g_idle_add(continue_replay, NULL);
    g_timer_start(clock_timer);
...

static gboolean
update_clock(gpointer dummy)
{
    char buf[10];
    sprintf(buf, "%d", (int)g_timer_elapsed(clock_timer, NULL));
    gtk_label_set_text(GTK_LABEL(clock_label), buf);
    return(TRUE);
}
*/
