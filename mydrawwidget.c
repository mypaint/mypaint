#include "mydrawwidget.h"
#include <gtk/gtkwidget.h>

static void mydrawwidget_class_init (MyDrawWidgetClass *klass);
static void mydrawwidget_init (MyDrawWidget *mdw);
static void mydrawwidget_destroy (GtkObject *object);

static GtkWidgetClass *parent_class = NULL;

GType
mydrawwidget_get_type (void)
{
  static GType mdw_type = 0;

  if (!mdw_type)
    {
      static const GTypeInfo mdw_info =
      {
	sizeof (MyDrawWidgetClass),
	NULL, /* base_init */
        NULL, /* base_finalize */
	(GClassInitFunc) mydrawwidget_class_init,
        NULL, /* class_finalize */
	NULL, /* class_data */
        sizeof (MyDrawWidget),
	0,
	(GInstanceInitFunc) mydrawwidget_init,
      };

      mdw_type = g_type_register_static (GTK_TYPE_DRAWING_AREA, "MyDrawWidget", &mdw_info, 0);
    }

  return mdw_type;
}

static gint
mydrawwidget_button_updown (GtkWidget *widget, GdkEventButton *event)
{
  MyDrawWidget * mdw;
  g_return_val_if_fail (widget != NULL, FALSE);
  g_return_val_if_fail (IS_MYDRAWWIDGET (widget), FALSE);
  mdw = MYDRAWWIDGET (widget);
  { // WARNING: code duplication, forced by different GdkEvent* structs.
    double pressure;
    if (!gdk_event_get_axis ((GdkEvent *)event, GDK_AXIS_PRESSURE, &pressure)) {
      pressure = (event->state & 256) ? 0.5 : 0;
    }
    //g_print ("motion %f %f %f %d\n", event->x, event->y, pressure, event->state);
    g_assert (pressure >= 0 && pressure <= 1);
    
    mdw->brush->queue_draw_widget = widget;
    brush_stroke_to (mdw->brush, mdw->surface, event->x, event->y, pressure, 
                     event->time / 1000.0 /* in seconds */ );
  } // END of duplicated code
  // TODO: actually react on button, if it was not triggered by pressure treshold
  return TRUE;
}

static gint
mydrawwidget_motion_notify (GtkWidget *widget, GdkEventMotion *event)
{
  MyDrawWidget * mdw;
  g_return_val_if_fail (widget != NULL, FALSE);
  g_return_val_if_fail (IS_MYDRAWWIDGET (widget), FALSE);
  mdw = MYDRAWWIDGET (widget);
  { // WARNING: code duplication, forced by different GdkEvent* structs.
    double pressure;
    if (!gdk_event_get_axis ((GdkEvent *)event, GDK_AXIS_PRESSURE, &pressure)) {
      pressure = (event->state & 256) ? 0.5 : 0;
    }
    //g_print ("motion %f %f %f %d\n", event->x, event->y, pressure, event->state);
    g_assert (pressure >= 0 && pressure <= 1);
    
    mdw->brush->queue_draw_widget = widget;
    brush_stroke_to (mdw->brush, mdw->surface, event->x, event->y, pressure, 
                     event->time / 1000.0 /* in seconds */ );
  } // END of duplicated code
  return TRUE;
}

gboolean
mydrawwidget_proximity_inout (GtkWidget *widget, GdkEventProximity *event)
{ 
  MyDrawWidget * mdw;

  g_return_val_if_fail (widget != NULL, FALSE);
  g_return_val_if_fail (IS_MYDRAWWIDGET (widget), FALSE);
  mdw = MYDRAWWIDGET (widget);

  g_print ("Proximity in/out: %s.\n", event->device->name);
  // TODO: change brush...
  // note, event is not received if it does not happen in our window,
  // so the motion event might actually be the first one to see a new device
  // Stroke certainly finished now.
  brush_reset (mdw->brush);
  return FALSE;
}

static gint
mydrawwidget_expose (GtkWidget *widget, GdkEventExpose *event)
{
  MyDrawWidget * mdw;
  guchar *rgb;
  int rowstride;

  g_return_val_if_fail (widget != NULL, FALSE);
  g_return_val_if_fail (IS_MYDRAWWIDGET (widget), FALSE);
  g_return_val_if_fail (event != NULL, FALSE);

  mdw = MYDRAWWIDGET (widget);

  rowstride = event->area.width * 3;
  rowstride = (rowstride + 3) & -4; /* align to 4-byte boundary */
  rgb = g_new (guchar, event->area.height * rowstride);

  surface_render (mdw->surface,
                  rgb, rowstride,
                  event->area.x, event->area.y,
                  event->area.width, event->area.height);

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

static void
mydrawwidget_class_init (MyDrawWidgetClass *class)
{
  GtkObjectClass *object_class;
  GtkWidgetClass *widget_class;

  object_class = (GtkObjectClass*) class;
  widget_class = (GtkWidgetClass*) class;

  parent_class = gtk_type_class (gtk_widget_get_type ());

  object_class->destroy = mydrawwidget_destroy;
  widget_class->expose_event = mydrawwidget_expose;
  widget_class->motion_notify_event = mydrawwidget_motion_notify;
  widget_class->button_press_event = mydrawwidget_button_updown;
  widget_class->button_release_event = mydrawwidget_button_updown;
  widget_class->proximity_in_event = mydrawwidget_proximity_inout;
  widget_class->proximity_out_event = mydrawwidget_proximity_inout;
  //widget_class->size_request = gtk_dial_size_request;
  //widget_class->size_allocate = gtk_dial_size_allocate;
}

static void
mydrawwidget_init (MyDrawWidget *mdw)
{
  mdw->brush = NULL;
  mdw->surface = new_surface (SIZE, SIZE);
  surface_clear (mdw->surface);
}

static void
mydrawwidget_destroy (GtkObject *object)
{
  MyDrawWidget * mdw;
  g_return_if_fail (object != NULL);
  g_return_if_fail (IS_MYDRAWWIDGET (object));
  mdw = MYDRAWWIDGET (object);
  // seems to be called multiple times
  if (mdw->surface) {
    free_surface (mdw->surface);
    mdw->surface = NULL;
  }
  if (GTK_OBJECT_CLASS (parent_class)->destroy)
    (* GTK_OBJECT_CLASS (parent_class)->destroy) (object);
}

GtkWidget*
mydrawwidget_new ()
{
  GtkWidget * da;
  da = GTK_WIDGET ( gtk_type_new (mydrawwidget_get_type ()) );
  // hope that's the right place for this stuff
  gtk_widget_set_extension_events (da, GDK_EXTENSION_EVENTS_ALL);
  gtk_widget_set_events (da, GDK_EXPOSURE_MASK
			 | GDK_LEAVE_NOTIFY_MASK
			 | GDK_BUTTON_PRESS_MASK
                         | GDK_BUTTON_RELEASE
			 | GDK_POINTER_MOTION_MASK
			 | GDK_PROXIMITY_IN_MASK
			 | GDK_PROXIMITY_OUT_MASK
                         );
  gtk_drawing_area_size (GTK_DRAWING_AREA (da), 400, 250);
  return da;
}

void	       
mydrawwidget_clear (MyDrawWidget *mdw)
{
  surface_clear (mdw->surface);
  gtk_widget_draw (GTK_WIDGET (mdw), NULL);
}


void
mydrawwidget_set_brush (MyDrawWidget *mdw, Brush * brush)
{
  mdw->brush = brush;
}
