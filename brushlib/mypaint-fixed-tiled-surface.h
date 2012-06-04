#ifndef MYPAINTFIXEDTILEDSURFACE_H
#define MYPAINTFIXEDTILEDSURFACE_H

#include <glib.h>

G_BEGIN_DECLS

/**
 * MyPaintFixedTiledSurface:
 *
 * Simple #MyPaintTiledSurface subclass that implements a fixed sized #MyPaintSurface.
 * Only intended for testing and trivial use-cases, and to serve as an example of
 * how to implement a tiled surface subclass.
 */
typedef struct _MyPaintGeglTiledSurface MyPaintFixedTiledSurface;

MyPaintFixedTiledSurface *
mypaint_fixed_tiled_surface_new(int width, int height);

G_END_DECLS

#endif // MYPAINTFIXEDTILEDSURFACE_H
