#ifndef MYPAINTBENCHMARKSURFACE_H
#define MYPAINTBENCHMARKSURFACE_H

#include "mypaint-test-surface.h"

int
mypaint_benchmark_surface_run(int argc, char **argv,
                      MyPaintTestsSurfaceFactory surface_factory,
                      gchar *title, gpointer user_data);

#endif // MYPAINTBENCHMARKSURFACE_H
