#ifndef MYPAINTTESTSURFACE_H
#define MYPAINTTESTSURFACE_H

#include <mypaint-surface.h>
#include <mypaint-glib-compat.h>

G_BEGIN_DECLS

typedef MyPaintSurface * (*MyPaintTestsSurfaceFactory)(gpointer user_data);

int
mypaint_test_surface_run(int argc, char **argv,
                      MyPaintTestsSurfaceFactory surface_factory,
                      gchar *title, gpointer user_data);

G_END_DECLS

#endif // MYPAINTTESTSURFACE_H
