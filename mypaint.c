#include <gtk/gtk.h>
#include <gdk/gdkkeysyms.h>
#include <stdlib.h>
#include <math.h>
#include "surface.h"
#include "brush.h"
#include "gtkmydrawwidget.h"

Brush * global_brush;
GtkWidget * main_mdw;
GtkWidget * statusline;

static void
init_input (void)
{
  GList *tmp_list;
  GdkDevice *info;

  tmp_list = gdk_devices_list();

  info = NULL;
  while (tmp_list)
    {
      info = (GdkDevice *)tmp_list->data;
      /*
      g_print ("device: %s\n", info->name);
      */
      if (!g_strcasecmp (info->name, "wacom") ||
	  !g_strcasecmp (info->name, "stylus") ||
	  !g_strcasecmp (info->name, "eraser"))
	  {
	    gdk_device_set_mode (info, GDK_MODE_SCREEN);
	  }
      tmp_list = tmp_list->next;
    }
  if (!info) return;
}

static void
brush_bigger (GtkAction *action, GtkWidget *window)
{
  //global_brush->radius *= 1.4;
}

static void
brush_smaller (GtkAction *action, GtkWidget *window)
{
  //global_brush->radius /= 1.4;
}

static void
invert_colors (GtkAction *action, GtkWidget *window)
{
  global_brush->color[0] = 255 - global_brush->color[0];
  global_brush->color[1] = 255 - global_brush->color[1];
  global_brush->color[2] = 255 - global_brush->color[2];

  /*
  global_brush->color[0] = g_random_int_range (0, 256);
  global_brush->color[1] = g_random_int_range (0, 256);
  global_brush->color[2] = g_random_int_range (0, 256);
  */
}

static void
clear_image (GtkAction *action, GtkWidget *window)
{
  gtk_my_draw_widget_clear (GTK_MY_DRAW_WIDGET (main_mdw));
}

static void
train_nn (GtkAction *action, GtkWidget *window)
{
  //neural_train ();
}

static GtkActionEntry my_actions[] = {
  { "ImageMenu", NULL, "Image" },
  { "BrushMenu", NULL, "Brush" },
  { "LearnMenu", NULL, "Learn" },
  { "ClearImage", NULL, "Clear", "<control>E", "Clear the image", G_CALLBACK (clear_image) },
  /*  { "Quit", NULL, "Quit", "<control>Q", "XXXX", G_CALLBACK (quit) }, */
  { "BrushBigger", NULL, "Bigger", "F", NULL, G_CALLBACK (brush_bigger) },
  { "BrushSmaller", NULL, "Smaller", "D", NULL, G_CALLBACK (brush_smaller) },
  { "InvertColor", NULL, "Invert Color", "X", NULL, G_CALLBACK (invert_colors) },
  { "TrainNN", NULL, "Train", "<control>T", NULL, G_CALLBACK (train_nn) },
};

/*
static GtkToggleActionEntry my_toggle_actions[] = {
  { "RecordData", NULL, "Record Data", "F11", "Record your inputs for later training", G_CALLBACK (nothing), TRUE },
  { "SuggestSize", NULL, "Suggest Size", "F12", "Foobar", G_CALLBACK (nothing), FALSE },
};
*/

static const char * ui_description = 
"<ui>"
"  <menubar name='MainMenu'>"
"    <menu action='ImageMenu'>"
"      <menuitem action='ClearImage' />"
"      <menuitem action='Quit' />"
"    </menu>"
"    <menu action='BrushMenu'>"
"      <menuitem action='BrushBigger' />"
"      <menuitem action='BrushSmaller' />"
"      <menuitem action='InvertColor' />"
"    </menu>"
"    <menu action='LearnMenu'>"
/*
"      <menuitem action='RecordData' />"
"      <menuitem action='SuggestSize' />"
*/
"      <menuitem action='TrainNN' />"
"    </menu>"
"  </menubar>"
"</ui>"
;

static void
on_hscale_brushoption_value_changed (GtkRange *range, gpointer user_data)
{
  int id;
  id = GPOINTER_TO_INT (user_data);
  brush_set_setting (global_brush, id, gtk_range_get_value (range));
}

void
create_brushsettings_window ()
{
  GtkWidget *window_brushoptions;
  GtkWidget *table1;
  GtkWidget *hscale;
  GtkWidget *label;
  GtkWidget *eventbox;
  int num_settings, i;
  BrushSettingInfo * setting;
  GtkTooltips *tooltips;

  tooltips = gtk_tooltips_new ();

  for (i=0; brush_setting_infos[i].cname; i++) ;
  num_settings = i;

  window_brushoptions = gtk_window_new (GTK_WINDOW_TOPLEVEL);
  gtk_window_set_title (GTK_WINDOW (window_brushoptions), "Brush settings");

  table1 = gtk_table_new (i, 2, FALSE);

  gtk_container_add (GTK_CONTAINER (window_brushoptions), table1);
  gtk_container_set_border_width (GTK_CONTAINER (table1), 4);
  gtk_table_set_col_spacings (GTK_TABLE (table1), 15);

  for (i=0; i<num_settings; i++) {
    setting = &brush_setting_infos[i];
    hscale = gtk_hscale_new (GTK_ADJUSTMENT (gtk_adjustment_new (setting->default_value, setting->min, setting->max, 0, 0, 0)));
    gtk_widget_set_size_request (hscale, 80, -1);
    gtk_table_attach (GTK_TABLE (table1), hscale, 1, 2, i, i+1,
                      (GtkAttachOptions) (GTK_EXPAND | GTK_FILL),
                      (GtkAttachOptions) (GTK_EXPAND | GTK_FILL), 0, 0);

    eventbox = gtk_event_box_new ();
    gtk_table_attach (GTK_TABLE (table1), eventbox, 0, 1, i, i+1,
                      (GtkAttachOptions) (GTK_FILL),
                      (GtkAttachOptions) (GTK_FILL), 0, 0);
    gtk_tooltips_set_tip (tooltips, eventbox, setting->helptext, NULL);

    label = gtk_label_new (setting->name);
    gtk_misc_set_alignment (GTK_MISC (label), 0, 0.5);
    gtk_container_add (GTK_CONTAINER (eventbox), label);
    
    g_signal_connect ((gpointer) hscale, "value_changed",
                      G_CALLBACK (on_hscale_brushoption_value_changed),
                      GINT_TO_POINTER (i) );

  }

  gtk_window_set_default_size (GTK_WINDOW (window_brushoptions), 400, -1);
  gtk_widget_show_all (window_brushoptions);
}

int
main (int argc, char **argv)
{
  GtkWidget *w;
  GtkWidget *v;

  gtk_init (&argc, &argv);

  init_input ();

  gdk_rgb_init ();

  gtk_widget_set_default_colormap (gdk_rgb_get_cmap ());
  gtk_widget_set_default_visual (gdk_rgb_get_visual ());

  w = gtk_window_new (GTK_WINDOW_TOPLEVEL);
  gtk_signal_connect (GTK_OBJECT (w), "destroy",
		      (GtkSignalFunc) gtk_main_quit, NULL);

  v = gtk_vbox_new (FALSE, 2);
  gtk_container_add (GTK_CONTAINER (w), v);
  gtk_widget_show (v);

  { /* Menu */
    GtkUIManager *uim;
    GError * error;
    GtkAccelGroup *accel_group;
    GtkActionGroup *action_group;
    GtkWidget *menu_bar;

    uim = gtk_ui_manager_new ();
    error = NULL;
    gtk_ui_manager_add_ui_from_string (uim, ui_description, -1, &error);
    if (error) 
      {
        g_print ("ERROR: %s\n", error->message);
        g_clear_error (&error);
        exit (1);
      }

    action_group = gtk_action_group_new ("myactiongroup");
    gtk_action_group_add_actions (action_group, my_actions, G_N_ELEMENTS (my_actions), w);
    /*
    gtk_action_group_add_toggle_actions (action_group, my_toggle_actions, G_N_ELEMENTS (my_toggle_actions), w);
    */
    gtk_ui_manager_insert_action_group (uim, action_group, 0);
    accel_group = gtk_ui_manager_get_accel_group (uim);
    gtk_window_add_accel_group (GTK_WINDOW (w), accel_group);

    menu_bar = gtk_ui_manager_get_widget (uim, "/MainMenu");
    /*
    recorddata = GTK_CHECK_MENU_ITEM (gtk_ui_manager_get_widget (uim, "/MainMenu/LearnMenu/RecordData"));
    suggestsize = GTK_CHECK_MENU_ITEM (gtk_ui_manager_get_widget (uim, "/MainMenu/LearnMenu/SuggestSize"));
    */
    gtk_box_pack_start (GTK_BOX (v), menu_bar, FALSE, FALSE, 0);

    gtk_accel_map_load ("accelmap.conf");
  }

  // main drawing widget (main_mdw)
  global_brush = brush_create ();
  main_mdw = gtk_my_draw_widget_new ();
  gtk_my_draw_widget_set_brush (GTK_MY_DRAW_WIDGET (main_mdw), global_brush); 
  gtk_box_pack_start (GTK_BOX (v), main_mdw, TRUE, TRUE, 0);

  statusline = gtk_label_new ("hello world");
  gtk_box_pack_start (GTK_BOX (v), statusline, FALSE, FALSE, 0);

  gtk_widget_show_all (w);

  create_brushsettings_window ();

  /*
  gtk_timeout_add (50, (GtkFunction) dry_timer, da);
  */

  gtk_main ();

  gtk_accel_map_save ("accelmap.conf");

  return 0;
}
