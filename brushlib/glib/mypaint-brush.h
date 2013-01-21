#ifndef MYPAINTBRUSHGLIB_H
#define MYPAINTBRUSHGLIB_H

#include <mypaint-config.h>

#if MYPAINT_CONFIG_USE_GLIB
#include <glib-object.h>

#define MYPAINT_TYPE_BRUSH               (mypaint_brush_get_type ())
#define MYPAINT_VALUE_HOLDS_BRUSH(value) (G_TYPE_CHECK_VALUE_TYPE ((value), MYPAINT_TYPE_BRUSH))
GType mypaint_brush_get_type(void);

#define MYPAINT_TYPE_SURFACE               (mypaint_surface_get_type ())
#define MYPAINT_VALUE_HOLDS_SURFACE(value) (G_TYPE_CHECK_VALUE_TYPE ((value), MYPAINT_TYPE_SURFACE))
GType mypaint_surface_get_type(void);

#define MYPAINT_TYPE_FIXED_TILED_SURFACE               (mypaint_fixed_tiled_surface_get_type ())
#define MYPAINT_VALUE_HOLDS_FIXED_TILED_SURFACE(value) (G_TYPE_CHECK_VALUE_TYPE ((value), MYPAINT_TYPE_FIXED_TILED_SURFACE))
GType mypaint_fixed_tiled_surface_get_type(void);

#define MYPAINT_TYPE_RECTANGLE               (mypaint_rectangle_get_type ())
#define MYPAINT_VALUE_HOLDS_RECTANGLE(value) (G_TYPE_CHECK_VALUE_TYPE ((value), MYPAINT_TYPE_RECTANGLE))
GType mypaint_rectangle_get_type(void);

#endif

#endif // MYPAINTBRUSHGLIB_H
