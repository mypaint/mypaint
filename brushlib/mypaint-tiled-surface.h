#ifndef MYPAINTTILEDSURFACE_H
#define MYPAINTTILEDSURFACE_H

#include <stdint.h>
#include <mypaint-surface.h>
#include <mypaint-config.h>

G_BEGIN_DECLS

struct _MyPaintTiledSurface;
typedef struct _MyPaintTiledSurface MyPaintTiledSurface;

typedef struct {
    int tx;
    int ty;
    gboolean readonly;
    guint16 *buffer;
    gpointer context; /* Only to be used by the surface implemenations. */
    int thread_id;
    int mipmap_level;
} MyPaintTileRequest;

void
mypaint_tile_request_init(MyPaintTileRequest *data, int level,
                          int tx, int ty, gboolean readonly);

typedef void (*MyPaintTileRequestStartFunction) (struct _MyPaintTiledSurface *self, MyPaintTileRequest *request);
typedef void (*MyPaintTileRequestEndFunction) (struct _MyPaintTiledSurface *self, MyPaintTileRequest *request);
typedef void (*MyPaintTiledSurfaceAreaChanged) (struct _MyPaintTiledSurface *self, int bb_x, int bb_y, int bb_w, int bb_h);

/**
  * MyPaintTiledSurface:
  *
  * Interface and convenience class for implementing a #MyPaintSurface backed by a tile store.
  *
  * The size of the surface is infinite, and consumers need just implement two vfuncs.
  */
struct _MyPaintTiledSurface {
    MyPaintSurface parent;
    /* private: */
    MyPaintTileRequestStartFunction tile_request_start;
    MyPaintTileRequestEndFunction tile_request_end;
    gboolean surface_do_symmetry;
    float surface_center_x;
    struct _OperationQueue *operation_queue;
    MyPaintRectangle dirty_bbox;
    gboolean threadsafe_tile_requests;
    int tile_size;
};

void
mypaint_tiled_surface_init(MyPaintTiledSurface *self,
                           MyPaintTileRequestStartFunction tile_request_start,
                           MyPaintTileRequestEndFunction tile_request_end);

void
mypaint_tiled_surface_destroy(MyPaintTiledSurface *self);

void
mypaint_tiled_surface_set_symmetry_state(MyPaintTiledSurface *self, gboolean active, float center_x);
float
mypaint_tiled_surface_get_alpha (MyPaintTiledSurface *self, float x, float y, float radius);

void mypaint_tiled_surface_tile_request_start(MyPaintTiledSurface *self, MyPaintTileRequest *request);
void mypaint_tiled_surface_tile_request_end(MyPaintTiledSurface *self, MyPaintTileRequest *request);

void mypaint_tiled_surface_begin_atomic(MyPaintTiledSurface *self);
void mypaint_tiled_surface_end_atomic(MyPaintTiledSurface *self, MyPaintRectangle *roi);

G_END_DECLS

#endif // MYPAINTTILEDSURFACE_H
