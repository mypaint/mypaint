/* brushlib - The MyPaint Brush Library
 * Copyright (C) 2012 Jon Nordby <jononor@gmail.com>
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#include <stdlib.h>
#include <assert.h>

#include "tilemap.h"

TileMap *
tile_map_new(int size, size_t item_size, TileMapItemFreeFunc item_free_func)
{
    TileMap *self = (TileMap *)malloc(sizeof(TileMap));

    self->size = size;
    self->item_size = item_size;
    self->item_free_func = item_free_func;
    const int map_size = 2*self->size*2*self->size;
    self->map = malloc(map_size*self->item_size);
    for(int i = 0; i < map_size; i++) {
        self->map[i] = NULL;
    }

    return self;
}

void
tile_map_free(TileMap *self, gboolean free_items)
{
    const int map_size = 2*self->size*2*self->size;
    if (free_items) {
        for(int i = 0; i < map_size; i++) {
            self->item_free_func(self->map[i]);
        }
    }
    free(self->map);

    free(self);
}

/* Get the data in the tile map for a given tile @index.
 * Must be reentrant and lock-free on different @index */
void **
tile_map_get(TileMap *self, TileIndex index)
{
    const int rowstride = self->size*2;
    const int offset = ((self->size + index.y) * rowstride) + self->size + index.x;
    assert(offset < 2*self->size*2*self->size);
    assert(offset >= 0);
    return self->map + offset;
}

/* Copy
 * The size of @other must be equal or larger to that of @self */
void
tile_map_copy_to(TileMap *self, TileMap *other)
{
    assert(other->size >= self->size);

    for(int y = -self->size; y < self->size; y++) {
        for(int x = -self->size; x < self->size; x++) {
            TileIndex index = {x, y};
            *tile_map_get(other, index) = *tile_map_get(self, index);
        }
    }
}

/* Must be reentrant and lock-free on different @index */
gboolean
tile_map_contains(TileMap *self, TileIndex index)
{
    return (index.x >= -self->size && index.x < self->size
            && index.y >= -self->size && index.y < self->size);
}

