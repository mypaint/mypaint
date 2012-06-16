
#include <mypaint-fixed-tiled-surface.h>
#include "mypaint-benchmark-surface.h"

#include <stddef.h>

MyPaintSurface *
fixed_surface_factory(gpointer user_data)
{
    MyPaintFixedTiledSurface * surface = mypaint_fixed_tiled_surface_new(1000, 1000);
    return (MyPaintSurface *)surface;
}

int
main(int argc, char **argv)
{
    return mypaint_benchmark_surface_run(argc, argv,
                                         fixed_surface_factory,"MyPaintFixedSurface", NULL);
}

