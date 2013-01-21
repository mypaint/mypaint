#include <mypaint-config.h>

#if MYPAINT_CONFIG_USE_GLIB

#include <glib-object.h>
#include <stdio.h>

GType
mypaint_gegl_tiled_surface_get_type (void)
{
    static GType type = 0;

    if (!type) {
        type = g_boxed_type_register_static("MyPaintGeglTiledSurface",
                                            (GBoxedCopyFunc) mypaint_surface_ref,
                                            (GBoxedFreeFunc) mypaint_surface_unref);
    }

    return type;
}

#endif
