#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include <mypaint-fixed-tiled-surface.h>


struct _MyPaintFixedTiledSurface {
    MyPaintTiledSurface parent;

    size_t tile_size; // Size (in bytes) of single tile
    uint16_t *tile_buffer; // Stores tiles in a linear chunk of memory (16bpc RGBA)
    uint16_t *null_tile; // Single tile that we hand out and ignore writes to
    int tiles_width; // width in tiles
    int tiles_height; // height in tiles
    int width; // width in pixels
    int height; // height in pixels

};

void free_simple_tiledsurf(MyPaintSurface *surface);

void reset_null_tile(MyPaintFixedTiledSurface *self)
{
    memset(self->null_tile, 0, self->tile_size);
}

static void
tile_request_start(MyPaintTiledSurface *tiled_surface, MyPaintTileRequest *request)
{
    MyPaintFixedTiledSurface *self = (MyPaintFixedTiledSurface *)tiled_surface;

    const int tx = request->tx;
    const int ty = request->ty;

    uint16_t *tile_pointer = NULL;

    if (tx >= self->tiles_width || ty >= self->tiles_height || tx < 0 || ty < 0) {
        // Give it a tile which we will ignore writes to
        tile_pointer = self->null_tile;

    } else {
        // Compute the offset for the tile into our linear memory buffer of tiles
        size_t rowstride = self->tiles_width * self->tile_size;
        size_t x_offset = tx * self->tile_size;
        size_t tile_offset = (rowstride * ty) + x_offset;

        tile_pointer = self->tile_buffer + tile_offset/sizeof(uint16_t);
    }

    request->buffer = tile_pointer;
}

static void
tile_request_end(MyPaintTiledSurface *tiled_surface, MyPaintTileRequest *request)
{
    MyPaintFixedTiledSurface *self = (MyPaintFixedTiledSurface *)tiled_surface;

    const int tx = request->tx;
    const int ty = request->ty;

    if (tx >= self->tiles_width || ty >= self->tiles_height || tx < 0 || ty < 0) {
        // Wipe any changed done to the null tile
        reset_null_tile(self);
    } else {
        // We hand out direct pointers to our buffer, so for the normal case nothing needs to be done
    }
}

MyPaintSurface *
mypaint_fixed_tiled_surface_interface(MyPaintFixedTiledSurface *self)
{
    return (MyPaintSurface *)self;
}

int
mypaint_fixed_tiled_surface_get_width(MyPaintFixedTiledSurface *self)
{
    return self->width;
}

int
mypaint_fixed_tiled_surface_get_height(MyPaintFixedTiledSurface *self)
{
    return self->height;
}

MyPaintFixedTiledSurface *
mypaint_fixed_tiled_surface_new(int width, int height)
{
    assert(width > 0);
    assert(height > 0);

    MyPaintFixedTiledSurface *self = (MyPaintFixedTiledSurface *)malloc(sizeof(MyPaintFixedTiledSurface));

    mypaint_tiled_surface_init(&self->parent, tile_request_start, tile_request_end);

    const int tile_size_pixels = self->parent.tile_size;

    // MyPaintSurface vfuncs
    self->parent.parent.destroy = free_simple_tiledsurf;

    const int tiles_width = ceil((float)width / tile_size_pixels);
    const int tiles_height = ceil((float)height / tile_size_pixels);
    const size_t tile_size = tile_size_pixels * tile_size_pixels * 4 * sizeof(uint16_t);
    const size_t buffer_size = tiles_width * tiles_height * tile_size;

    assert(tile_size_pixels*tiles_width >= width);
    assert(tile_size_pixels*tiles_height >= height);
    assert(buffer_size >= width*height*4*sizeof(uint16_t));

    uint16_t * buffer = (uint16_t *)malloc(buffer_size);
    if (!buffer) {
        fprintf(stderr, "CRITICAL: unable to allocate enough memory: %Zu bytes", buffer_size);
        return NULL;
    }
    memset(buffer, 255, buffer_size);

    self->tile_buffer = buffer;
    self->tile_size = tile_size;
    self->null_tile = (uint16_t *)malloc(tile_size);
    self->tiles_width = tiles_width;
    self->tiles_height = tiles_height;
    self->height = height;
    self->width = width;

    reset_null_tile(self);

    return self;
}

void free_simple_tiledsurf(MyPaintSurface *surface)
{
    MyPaintFixedTiledSurface *self = (MyPaintFixedTiledSurface *)surface;

    mypaint_tiled_surface_destroy(&self->parent);

    free(self->tile_buffer);
    free(self->null_tile);

    free(self);
}

