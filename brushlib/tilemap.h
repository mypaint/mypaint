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

#ifndef TILEMAP_H
#define TILEMAP_H

#include <mypaint-glib-compat.h>

G_BEGIN_DECLS

typedef struct {
    int x;
    int y;
} TileIndex;

typedef void (*TileMapItemFreeFunc) (void *item_data);

// A size of 10 means the map spans x=[-10,9], y=[-10,9]
// The tile with TileIndex (x,y) is stored in the map at offset
// offset=((self->size + y) * rowstride) + (self->size + index.x)
typedef struct {
    void **map;
    int size;
    size_t item_size;
    TileMapItemFreeFunc item_free_func;
} TileMap;

TileMap *
tile_map_new(int size, size_t item_size, TileMapItemFreeFunc item_free_func);

void
tile_map_free(TileMap *self, gboolean free_items);

gboolean
tile_map_contains(TileMap *self, TileIndex index);

void **
tile_map_get(TileMap *self, TileIndex index);

void
tile_map_copy_to(TileMap *self, TileMap *other);

G_END_DECLS

#endif // TILEMAP_H
