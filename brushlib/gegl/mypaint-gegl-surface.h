#ifndef MYPAINTGEGLSURFACE_H
#define MYPAINTGEGLSURFACE_H

#include <gegl.h>

typedef struct _MyPaintGeglTiledSurface MyPaintGeglTiledSurface;

GeglBuffer *
mypaint_gegl_tiled_surface_get_buffer(MyPaintGeglTiledSurface *self);

void
mypaint_gegl_tiled_surface_set_buffer(MyPaintGeglTiledSurface *self, GeglBuffer *buffer);

MyPaintGeglTiledSurface *
mypaint_gegl_tiled_surface_new();

#endif // MYPAINTGEGLSURFACE_H
