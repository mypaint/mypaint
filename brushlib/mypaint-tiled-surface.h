#ifndef MYPAINTTILEDSURFACE_H
#define MYPAINTTILEDSURFACE_H

#include <mypaint-surface.h>

typedef struct _MyPaintTiledSurface MyPaintTiledSurface;

typedef uint16_t *(*MyPaintTiledSurfaceGetTileFunction) (MyPaintTiledSurface *self, int tx, int ty, bool readonly);
typedef void (*MyPaintTiledSurfaceUpdateTileFunction) (MyPaintTiledSurface *self, int tx, int ty, uint16_t * tile_buffer);
typedef void (*MyPaintTiledSurfaceAtomicChangeFunction) (MyPaintTiledSurface *self);

/**
  * MyPaintTiledSurface:
  *
  * MyPaintSurface backed by a tile store. The size of the surface is infinite.
  */

/**
  * mypaint_tiled_surface_new:
  *
  * Create a new MyPaintTiledSurface.
  */
MyPaintTiledSurface *
mypaint_tiled_surface_new(PyObject *);

void
mypaint_tiled_surface_set_symmetry_state(MyPaintTiledSurface *self, bool active, float center_x);
float
mypaint_tiled_surface_get_alpha (MyPaintTiledSurface *self, float x, float y, float radius);

void mypaint_tiled_surface_begin_atomic(MyPaintTiledSurface *self);
void mypaint_tiled_surface_end_atomic(MyPaintTiledSurface *self);
uint16_t * mypaint_tiled_surface_get_tile(MyPaintTiledSurface *self, int tx, int ty, bool readonly);
void mypaint_tiled_surface_update_tile(MyPaintTiledSurface *self, int tx, int ty, uint16_t * tile_buffer);

#endif // MYPAINTTILEDSURFACE_H
