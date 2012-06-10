
#include <mypaint-config.h>

#if MYPAINT_CONFIG_USE_GLIB

#include <glib-object.h>

MyPaintBrush *
mypaint_brush_copy (MyPaintBrush *brush)
{
    // XXX: should this be a deep copy?
    return (MyPaintBrush *) g_memdup (brush, sizeof (MyPaintBrush));
}

GType
mypaint_brush_get_type (void)
{
    static GType brush_type = 0;

    if (!brush_type) {
        brush_type = g_boxed_type_register_static ("MyPaintBrush",
                                               (GBoxedCopyFunc) mypaint_brush_copy,
                                               (GBoxedFreeFunc) mypaint_brush_destroy);
    }

    return brush_type;
}

#endif
