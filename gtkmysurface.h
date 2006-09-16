#ifndef __GTK_MY_SURFACE_H__
#define __GTK_MY_SURFACE_H__

#include <glib.h>
#include <glib-object.h>
#include <gdk/gdk.h>

#define GTK_TYPE_MY_SURFACE            (gtk_my_surface_get_type ())
#define GTK_MY_SURFACE(obj)            (G_TYPE_CHECK_INSTANCE_CAST ((obj), GTK_TYPE_MY_SURFACE, GtkMySurface))
#define GTK_MY_SURFACE_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST ((klass), GTK_TYPE_MY_SURFACE, GtkMySurfaceClass))
#define GTK_IS_MY_SURFACE(obj)         (G_TYPE_CHECK_INSTANCE_TYPE ((obj), GTK_TYPE_MY_SURFACE))
#define GTK_IS_MY_SURFACE_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE ((klass), GTK_TYPE_MY_SURFACE))
#define GTK_MY_SURFACE_GET_CLASS(obj)  (G_TYPE_INSTANCE_GET_CLASS ((obj), GTK_TYPE_MY_SURFACE, GtkMySurfaceClass))


typedef struct _GtkMySurface       GtkMySurface;
typedef struct _GtkMySurfaceClass  GtkMySurfaceClass;

// virtual class, common parent // currently does nothing.
struct _GtkMySurface
{
  GObject parent;
};

struct _GtkMySurfaceClass
{
  GObjectClass parent_class;

  // virtual functions:
  void (*clear) (GtkMySurface *s);

  // any emitted signals? notification of change maxrects?
  //void (*dragging_finished) (GtkMySurface *mdw);
};

GType      gtk_my_surface_get_type   (void) G_GNUC_CONST;

GtkMySurface* gtk_my_surface_new        (void);

void gtk_my_surface_clear (GtkMySurface *s);

#endif /* __GTK_MY_SURFACE_H__ */
