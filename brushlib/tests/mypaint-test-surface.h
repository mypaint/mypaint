#ifndef MYPAINTTESTSURFACE_H
#define MYPAINTTESTSURFACE_H

#include <mypaint-surface.h>

typedef MyPaintSurface * (*MyPaintTestsSurfaceFactory)(gpointer user_data);

int
mypaint_test_surface_run(int argc, char **argv,
                      MyPaintTestsSurfaceFactory surface_factory,
                      gchar *title, gpointer user_data);

#endif // MYPAINTTESTSURFACE_H
