#ifndef MYPAINTTILEDSURFACE_H
#define MYPAINTTILEDSURFACE_H

#include <stdint.h>
#include <mypaint-surface.h>

G_BEGIN_DECLS

struct _MyPaintTiledSurface;
typedef struct _MyPaintTiledSurface MyPaintTiledSurface;

typedef struct {
    int tx;
    int ty;
    gboolean readonly;
    guint16 *buffer;
    gpointer context; /* Only to be used by the surface implemenations. */
} MyPaintTiledSurfaceTileRequestData;

void
mypaint_tiled_surface_tile_request_init(MyPaintTiledSurfaceTileRequestData *data,
                                        int tx, int ty, gboolean readonly);

typedef void (*MyPaintTiledSurfaceTileRequestStartFunction) (struct _MyPaintTiledSurface *self, MyPaintTiledSurfaceTileRequestData *request);
typedef void (*MyPaintTiledSurfaceTileRequestEndFunction) (struct _MyPaintTiledSurface *self, MyPaintTiledSurfaceTileRequestData *request);
typedef void (*MyPaintTiledSurfaceAreaChanged) (struct _MyPaintTiledSurface *self, int bb_x, int bb_y, int bb_w, int bb_h);

typedef struct _OperationQueue OperationQueue;

/**
  * MyPaintTiledSurface:
  *
  * MyPaintSurface backed by a tile store. The size of the surface is infinite.
  */
struct _MyPaintTiledSurface {
    MyPaintSurface parent;
    MyPaintTiledSurfaceTileRequestStartFunction tile_request_start;
    MyPaintTiledSurfaceTileRequestEndFunction tile_request_end;
    MyPaintTiledSurfaceAreaChanged area_changed;

    /* private: */
    gboolean surface_do_symmetry;
    float surface_center_x;
    OperationQueue *operation_queue;
};


/**
  * mypaint_tiled_surface_new:
  *
  * Create a new MyPaintTiledSurface.
  */
void
mypaint_tiled_surface_init(MyPaintTiledSurface *);

void
mypaint_tiled_surface_destroy(MyPaintTiledSurface *self);

void
mypaint_tiled_surface_set_symmetry_state(MyPaintTiledSurface *self, gboolean active, float center_x);
float
mypaint_tiled_surface_get_alpha (MyPaintTiledSurface *self, float x, float y, float radius);

void mypaint_tiled_surface_tile_request_start(MyPaintTiledSurface *self, MyPaintTiledSurfaceTileRequestData *request);
void mypaint_tiled_surface_tile_request_end(MyPaintTiledSurface *self, MyPaintTiledSurfaceTileRequestData *request);

void mypaint_tiled_surface_area_changed(MyPaintTiledSurface *self, int bb_x, int bb_y, int bb_w, int bb_h);

void mypaint_tiled_surface_begin_atomic(MyPaintTiledSurface *self);
void mypaint_tiled_surface_end_atomic(MyPaintTiledSurface *self);

G_END_DECLS

#endif // MYPAINTTILEDSURFACE_H
