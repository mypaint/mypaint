
#include <stddef.h>

#include <mypaint-fixed-tiled-surface.h>
#include "mypaint-test-surface.h"

MyPaintSurface *
fixed_surface_factory(gpointer user_data)
{
    MyPaintFixedTiledSurface * surface = mypaint_fixed_tiled_surface_new(1000, 1000);
    return (MyPaintSurface *)surface;
}

int
main(int argc, char **argv)
{
    return mypaint_test_surface_run(argc, argv, fixed_surface_factory, "MyPaintFixedSurface", NULL);
}

