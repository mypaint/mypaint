
#include <stddef.h>

#include <mypaint-gegl-surface.h>
#include "mypaint-test-surface.h"

MyPaintSurface *
gegl_surface_factory(gpointer user_data)
{
    MyPaintGeglTiledSurface * surface = mypaint_gegl_tiled_surface_new();
    return (MyPaintSurface *)surface;
}

int
main(int argc, char **argv)
{
    babl_init();
    gegl_init(0, NULL);

    int retval = mypaint_test_surface_run(argc, argv, gegl_surface_factory, "MyPaintGeglSurface", NULL);

    gegl_exit();
    return retval;
}

