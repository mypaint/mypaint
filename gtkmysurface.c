/* This file is part of MyPaint.
 * Copyright (C) 2007 by Martin Renold <martinxyz@gmx.ch>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License.
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY. See the COPYING file for more details.
 */

#include <stdio.h>
#include <string.h>
#include <glib.h>
#include <math.h>
#include "gtkmysurface.h"
#include "helpers.h"
#include "mymarshal.h"

static void gtk_my_surface_class_init    (GtkMySurfaceClass *klass);
static void gtk_my_surface_init          (GtkMySurface      *b);
static void gtk_my_surface_finalize (GObject *object);

static gpointer parent_class;

enum {
  MODIFIED,
  LAST_SIGNAL
};
guint gtk_my_surface_signals[LAST_SIGNAL] = { 0 };

GType
gtk_my_surface_get_type (void)
{
  static GType type = 0;

  if (!type)
    {
      static const GTypeInfo info =
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

      type =
	g_type_register_static (G_TYPE_OBJECT, "GtkMySurface",
				&info, 0);
    }

  return type;
}

static void
gtk_my_surface_class_init (GtkMySurfaceClass *class)
{
  GObjectClass *gobject_class = G_OBJECT_CLASS (class);
  parent_class = g_type_class_peek_parent (class);
  gobject_class->finalize = gtk_my_surface_finalize;

  class->clear = NULL; // pure virtual method

  gtk_my_surface_signals[MODIFIED] = g_signal_new 
    ("surface_modified",
     G_TYPE_FROM_CLASS (class),
     G_SIGNAL_RUN_LAST,
     G_STRUCT_OFFSET (GtkMySurfaceClass, surface_modified),
     NULL, NULL,
     mymarshal_VOID__INT_INT_INT_INT,
     G_TYPE_NONE, 4,
     G_TYPE_INT,
     G_TYPE_INT,
     G_TYPE_INT,
     G_TYPE_INT);
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
  //g_print ("gtk_my_surface_new (This should get called.)\n");
  return g_object_new (GTK_TYPE_MY_SURFACE, NULL);
}

void gtk_my_surface_clear (GtkMySurface *s)
{
  GTK_MY_SURFACE_GET_CLASS(s)->clear (s);
}

void
gtk_my_surface_modified (GtkMySurface *s, gint x, gint y, gint w, gint h)
{
  g_return_if_fail (GTK_IS_MY_SURFACE (s));
  g_signal_emit (s, gtk_my_surface_signals[MODIFIED], 0, x, y, w, h);
}
