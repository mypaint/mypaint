
#include <mypaint-config.h>

#if MYPAINT_CONFIG_USE_GLIB

#include <glib-object.h>

GType
mypaint_brush_get_type (void)
{
    static GType type = 0;

    if (!type) {
        type = g_boxed_type_register_static("MyPaintBrush",
                                            (GBoxedCopyFunc) mypaint_brush_ref,
                                            (GBoxedFreeFunc) mypaint_brush_unref);
    }

    return type;
}

GType
mypaint_surface_get_type (void)
{
    static GType type = 0;

    if (!type) {
        type = g_boxed_type_register_static("MyPaintSurface",
                                            (GBoxedCopyFunc) mypaint_surface_ref,
                                            (GBoxedFreeFunc) mypaint_surface_unref);
    }

    return type;
}

GType
mypaint_fixed_tiled_surface_get_type (void)
{
    static GType type = 0;

    if (!type) {
        type = g_boxed_type_register_static("MyPaintFixedTiledSurface",
                                            (GBoxedCopyFunc) mypaint_surface_ref,
                                            (GBoxedFreeFunc) mypaint_surface_unref);
    }

    return type;
}

GType
mypaint_rectangle_get_type(void)
{
    static GType type = 0;

    if (!type) {
        type = g_boxed_type_register_static("MyPaintRectangle",
                                            (GBoxedCopyFunc) mypaint_rectangle_copy,
                                            (GBoxedFreeFunc) free);
    }

    return type;
}

#endif
