#include <stdio.h>
#include <string.h>
#include <glib.h>
#include <math.h>
#include "gtkmysurface.h"
#include "helpers.h"

static void gtk_my_surface_class_init    (GtkMySurfaceClass *klass);
static void gtk_my_surface_init          (GtkMySurface      *b);
static void gtk_my_surface_finalize (GObject *object);

static gpointer parent_class;

GType
gtk_my_surface_get_type (void)
{
  static GType my_surface_type = 0;

  if (!my_surface_type)
    {
      static const GTypeInfo my_surface_info =
      {
	sizeof (GtkMySurfaceClass),
	NULL,		/* base_init */
	NULL,		/* base_finalize */
	(GClassInitFunc) gtk_my_surface_class_init,
	NULL,		/* class_finalize */
	NULL,		/* class_data */
	sizeof (GtkMySurface),
	0,		/* n_preallocs */
	(GInstanceInitFunc) gtk_my_surface_init,
      };

      my_surface_type =
	g_type_register_static (G_TYPE_OBJECT, "GtkMySurface",
				&my_surface_info, 0);
    }

  return my_surface_type;
}

static void
gtk_my_surface_class_init (GtkMySurfaceClass *class)
{
  GObjectClass *gobject_class = G_OBJECT_CLASS (class);
  parent_class = g_type_class_peek_parent (class);
  gobject_class->finalize = gtk_my_surface_finalize;

  class->clear = NULL; // pure virtual method
}

static void
gtk_my_surface_init (GtkMySurface *b)
{
  // nothing to do
}

static void
gtk_my_surface_finalize (GObject *object)
{
  // nothing to do
  G_OBJECT_CLASS (parent_class)->finalize (object);
}

GtkMySurface*
gtk_my_surface_new (void)
{
  g_print ("gtk_my_surface_new (This should get called.)\n");
  return g_object_new (GTK_TYPE_MY_SURFACE, NULL);
}

void gtk_my_surface_clear (GtkMySurface *s)
{
  GTK_MY_SURFACE_GET_CLASS(s)->clear (s);
}
