#include <gtk/gtk.h>
#include <stdlib.h>
#include <math.h>
#include "surfacepaint.h"

Surface * screen; /* the global bitmap */
Brush brush; /* global brush */

double lastx, lasty;
double lastpaintx, lastpainty, lastpaintpress;
double dist;
double spacing = 0.2;

GtkWidget *statusline;

static gint
my_button_press (GtkWidget *widget, GdkEventButton *event)
{
  g_print ("button press %f %f %f\n", event->x, event->y, event->pressure);

  /*brush.radius = 3.0;*/
  /*
  brush.opaque = event->pressure;
  brush.radius = 3.0;
  brush.color[0] = 0;
  brush.color[1] = 0;
  brush.color[2] = 0;
  surface_draw (screen,
                event->x,
                event->y,
                &brush);

  lastx = event->x;
  lasty = event->y;
  dist = 0;
  */

  gtk_widget_queue_draw (widget);
  return TRUE;
}

static gint
my_motion (GtkWidget *widget, GdkEventMotion *event)
{
  double delta;
  /* g_print ("motion %f %f %f %d\n", event->x, event->y, event->pressure, event->state); */

  /* no button pressed */
  /* don't care, always paint - FIXME: breaks mouse support (always pressure set)
  if (!(event->state & 256))
    return TRUE;
  */

  delta = sqrt(sqr(event->x - lastx) + sqr(event->y - lasty));

  dist += delta;

  while (dist >= spacing*brush.radius)
    {
      /* interpolate position and presssure (a bit wrong probably, but works great so far) */
      lastpaintx = lastpaintx + (event->x - lastpaintx) * spacing*brush.radius / dist;
      lastpainty = lastpainty + (event->y - lastpainty) * spacing*brush.radius / dist;
      lastpaintpress = lastpaintpress + (event->pressure - lastpaintpress) * spacing*brush.radius / dist;
      dist -= spacing*brush.radius;
      /*
      if (lastpaintpress > 0.7) brush.radius *= 1.0 + (lastpaintpress-0.7)*0.03;
      if (lastpaintpress < 0.3 && lastpaintpress > 0.0) brush.radius /= 1.0 + (0.3-lastpaintpress)*0.03;
      if (brush.radius < 0.6) brush.radius = 0.6;
      if (brush.radius > 60) brush.radius = 60;
      */

      brush.opaque = lastpaintpress / 4.0;
      surface_draw (screen,
                    lastpaintx,
                    lastpainty,
                    &brush);

      gtk_widget_queue_draw_area (widget, 
                                  floor(lastpaintx-(brush.radius+1)),
                                  floor(lastpainty-(brush.radius+1)),
                                  /* FIXME: think about it exactly */
                                  ceil (2*(brush.radius+1)),
                                  ceil (2*(brush.radius+1))
                                  );
    }

  lastx = event->x;
  lasty = event->y;

  return TRUE;
}

static gint
my_expose (GtkWidget *widget, GdkEventExpose *event, Surface *s)
{
  byte *rgb;
  int rowstride;

  rowstride = event->area.width * 3;
  rowstride = (rowstride + 3) & -4; /* align to 4-byte boundary */
  rgb = g_new (byte, event->area.height * rowstride);

  surface_render (s,
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


static gint
clear_button_press (GtkWidget *widget, GtkWidget *da)
{
  surface_clear (screen);
  gtk_widget_draw (da, NULL);
  return TRUE;
}

static void
init_input (void)
{
  GList *tmp_list;
  GdkDeviceInfo *info;

  tmp_list = gdk_input_list_devices();

  info = NULL;
  while (tmp_list)
    {
      info = (GdkDeviceInfo *)tmp_list->data;
      /*
      g_print ("device: %s\n", info->name);
      */
      if (!g_strcasecmp (info->name, "wacom") ||
	  !g_strcasecmp (info->name, "stylus") ||
	  !g_strcasecmp (info->name, "eraser"))
	  {
	    gdk_input_set_mode (info->deviceid, GDK_MODE_SCREEN);
	  }
      tmp_list = tmp_list->next;
    }
  if (!info) return;
}

int
main (int argc, char **argv)
{
  GtkWidget *w;
  GtkWidget *v;
  GtkWidget *eb;
  GtkWidget *da;
  GtkWidget *h;
  GtkWidget *db;
  int xs = SIZE;
  int ys = SIZE;

  gtk_init (&argc, &argv);

  if (argc >= 3)
    {
      xs = atoi (argv[1]);
      ys = atoi (argv[2]);
      if (xs == 0) xs = SIZE;
      if (ys == 0) ys = SIZE;
    }


  init_input ();

  gdk_rgb_init ();

  gtk_widget_set_default_colormap (gdk_rgb_get_cmap ());
  gtk_widget_set_default_visual (gdk_rgb_get_visual ());

  screen = new_surface (xs, ys);

  brush.radius = 8.0;
  brush.color[0] = 0;
  brush.color[1] = 0;
  brush.color[2] = 0;

  w = gtk_window_new (GTK_WINDOW_TOPLEVEL);
  gtk_signal_connect (GTK_OBJECT (w), "destroy",
		      (GtkSignalFunc) gtk_main_quit, NULL);

  v = gtk_vbox_new (FALSE, 2);
  gtk_container_add (GTK_CONTAINER (w), v);
  gtk_widget_show (v);

  eb = gtk_event_box_new ();
  gtk_container_add (GTK_CONTAINER (v), eb);
  gtk_widget_show (eb);

  gtk_widget_set_extension_events (eb, GDK_EXTENSION_EVENTS_ALL);

  gtk_widget_set_events (eb, GDK_EXPOSURE_MASK
			 | GDK_LEAVE_NOTIFY_MASK
			 | GDK_BUTTON_PRESS_MASK
			 | GDK_KEY_PRESS_MASK
			 | GDK_POINTER_MOTION_MASK
			 | GDK_PROXIMITY_OUT_MASK);

  gtk_signal_connect (GTK_OBJECT (eb), "button_press_event",
		      (GtkSignalFunc) my_button_press, NULL);
  gtk_signal_connect (GTK_OBJECT (eb), "motion_notify_event",
		      (GtkSignalFunc) my_motion, NULL);

  da = gtk_drawing_area_new ();
  gtk_drawing_area_size (GTK_DRAWING_AREA (da), xs, ys);
  gtk_container_add (GTK_CONTAINER (eb), da);
  gtk_widget_show (da);

  gtk_signal_connect (GTK_OBJECT (da), "expose_event",
		      (GtkSignalFunc) my_expose, screen);

  statusline = gtk_label_new ("hello world");
  gtk_container_add (GTK_CONTAINER (v), statusline);
  gtk_widget_show (statusline);

  h = gtk_hbox_new (TRUE, 5);
  gtk_container_add (GTK_CONTAINER (v), h);
  gtk_widget_show (h);

  db = gtk_button_new_with_label ("Clear");
  gtk_container_add (GTK_CONTAINER (h), db);
  gtk_widget_show (db);
  gtk_signal_connect (GTK_OBJECT (db), "clicked",
		      (GtkSignalFunc) clear_button_press, da);

  gtk_widget_show (w);

  /*
  gtk_timeout_add (50, (GtkFunction) dry_timer, da);
  */

  gtk_main ();
  
  return 0;
}
