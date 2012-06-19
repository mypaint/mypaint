
#include <malloc.h>
#include <assert.h>

#include <mypaint-tiled-surface.h>
#include "mypaint-gegl-surface.h"
#include <gegl-utils.h>

#define TILE_SIZE 64

typedef struct _MyPaintGeglTiledSurface {
    MyPaintTiledSurface parent;

    int atomic;
    //Rect dirty_bbox; TODO: change into a GeglRectangle

    GeglRectangle extent_rect; // TODO: remove, just use the extent of the buffer
    GeglBuffer *buffer;
    const Babl *format;
    GeglBufferIterator *buffer_iterator;
} MyPaintGeglTiledSurface;

void free_gegl_tiledsurf(MyPaintSurface *surface);

void begin_atomic_gegl(MyPaintTiledSurface *tiled_surface)
{
    MyPaintGeglTiledSurface *self = (MyPaintGeglTiledSurface *)tiled_surface;

    if (self->atomic == 0) {
      //assert(self->dirty_bbox.w == 0);
    }
    self->atomic++;
}

void end_atomic_gegl(MyPaintTiledSurface *tiled_surface)
{
    MyPaintGeglTiledSurface *self = (MyPaintGeglTiledSurface *)tiled_surface;

    assert(self->atomic > 0);
    self->atomic--;

    if (self->atomic == 0) {
      //Rect bbox = self->dirty_bbox;
      //self->dirty_bbox.w = 0;
      //if (bbox.w > 0) {
         // TODO: Could notify of changes here instead of for each tile changed
      //}
    }
}

uint16_t *
get_tile_memory_gegl(MyPaintTiledSurface *tiled_surface, int tx, int ty, gboolean readonly)
{
    MyPaintGeglTiledSurface *self = (MyPaintGeglTiledSurface *)tiled_surface;

    GeglRectangle tile_bbox;
    gegl_rectangle_set(&tile_bbox, tx * TILE_SIZE, ty * TILE_SIZE, TILE_SIZE, TILE_SIZE);

    int read_write_flags;

    if (readonly) {
        read_write_flags = GEGL_BUFFER_READ;
    } else {
        read_write_flags = GEGL_BUFFER_READWRITE;

        // Extend the bounding box
        gegl_rectangle_bounding_box(&self->extent_rect, &self->extent_rect, &tile_bbox);
        gboolean success = gegl_buffer_set_extent(self->buffer, &self->extent_rect);
        g_assert(success);
    }

    uint16_t * tile_buffer = NULL;
    self->buffer_iterator = gegl_buffer_iterator_new(self->buffer, &tile_bbox, 0, self->format,
                                  read_write_flags, GEGL_ABYSS_NONE);

    // Read out
    gboolean completed = gegl_buffer_iterator_next(self->buffer_iterator);

    g_assert(completed);

    if (self->buffer_iterator->length != TILE_SIZE*TILE_SIZE) {
        g_critical("Unable to get tile aligned access to GeglBuffer");
        return NULL;
    } else {
        tile_buffer = (uint16_t *)(self->buffer_iterator->data[0]);
    }

    return tile_buffer;
}

void update_tile_gegl(MyPaintTiledSurface *tiled_surface, int tx, int ty, uint16_t * tile_buffer)
{
    MyPaintGeglTiledSurface *self = (MyPaintGeglTiledSurface *)tiled_surface;

    if (self->buffer_iterator) {
          gegl_buffer_iterator_next(self->buffer_iterator);
          self->buffer_iterator = NULL;
    }
}

void area_changed_gegl(MyPaintTiledSurface *tiled_surface, int bb_x, int bb_y, int bb_w, int bb_h)
{
    MyPaintGeglTiledSurface *self = (MyPaintGeglTiledSurface *)tiled_surface;

    // TODO: use gegl_rectangle_bounding_box instead
    //ExpandRectToIncludePoint (&self->dirty_bbox, bb_x, bb_y);
    //ExpandRectToIncludePoint (&self->dirty_bbox, bb_x+bb_w-1, bb_y+bb_h-1);
}

void
save_png(MyPaintSurface *surface, const char *path,
         int x, int y, int width, int height)
{
    MyPaintGeglTiledSurface *self = (MyPaintGeglTiledSurface *)surface;
    GeglNode *graph, *save, *source;

    graph = gegl_node_new();
    source = gegl_node_new_child(graph, "operation", "gegl:buffer-source",
                                 "buffer", mypaint_gegl_tiled_surface_get_buffer(self), NULL);
    save = gegl_node_new_child(graph, "operation", "gegl:png-save", "path", path, NULL);
    gegl_node_link(source, save);

    gegl_node_process(save);
    g_object_unref(graph);
}

GeglBuffer *
mypaint_gegl_tiled_surface_get_buffer(MyPaintGeglTiledSurface *self)
{
    return self->buffer;
}

void
mypaint_gegl_tiled_surface_set_buffer(MyPaintGeglTiledSurface *self, GeglBuffer *buffer)
{
    if (buffer && self->buffer == buffer) {
        return;
    }

    if (self->buffer) {
        g_object_unref(self->buffer);
    }

    if (buffer) {
        g_return_if_fail(GEGL_IS_BUFFER(buffer));
        g_object_ref(buffer);
        self->buffer = buffer;
    } else {
        // Using GeglBuffer with aligned tiles for zero-copy access
        self->buffer = GEGL_BUFFER(g_object_new(GEGL_TYPE_BUFFER,
                          "x", self->extent_rect.x, "y", self->extent_rect.y,
                          "width", self->extent_rect.width, "height", self->extent_rect.height,
                          "format", self->format,
                          "tile-width", TILE_SIZE, "tile-height", TILE_SIZE,
                          NULL));
    }
    g_assert(GEGL_IS_BUFFER(self->buffer));
}

MyPaintGeglTiledSurface *
mypaint_gegl_tiled_surface_new()
{
    MyPaintGeglTiledSurface *self = (MyPaintGeglTiledSurface *)malloc(sizeof(MyPaintGeglTiledSurface));

    mypaint_tiled_surface_init(&self->parent);

    self->parent.parent.destroy = free_gegl_tiledsurf;
    self->parent.parent.save_png = save_png;

    self->parent.get_tile = get_tile_memory_gegl;
    self->parent.update_tile = update_tile_gegl;
    self->parent.begin_atomic = begin_atomic_gegl;
    self->parent.end_atomic = end_atomic_gegl;
    self->parent.area_changed = area_changed_gegl;

    self->atomic = 0;
    //self->dirty_bbox.w = 0;
    self->buffer_iterator = NULL;
    self->buffer = NULL;

    gegl_rectangle_set(&self->extent_rect, 0, 0, 0, 0);

    self->format = babl_format_new(babl_model ("R'aG'aB'aA"), babl_type ("u15"),
                             babl_component("R'a"), babl_component("G'a"), babl_component("B'a"), babl_component("A"),
                             NULL);
    g_assert(self->format);

    mypaint_gegl_tiled_surface_set_buffer(self, NULL);

    return self;
}

void free_gegl_tiledsurf(MyPaintSurface *surface)
{
    MyPaintGeglTiledSurface *self = (MyPaintGeglTiledSurface *)surface;

    mypaint_tiled_surface_destroy(&self->parent);
    g_object_unref(self->buffer);

    free(self);
}
