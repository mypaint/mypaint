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

#include <mypaint-glib-compat.h>
#include "operationqueue.h"
#include "fifo.h"

struct _OperationQueue {
    TileMap *tile_map;

    TileIndex *dirty_tiles;
    int dirty_tiles_n;
};

/* For use with queue_delete */
void
operation_delete_func(void *user_data) {
    if (user_data) {
        free(user_data);
    }
}


void
free_fifo(void *item) {
    Fifo *op_queue = item;
    if (op_queue) {
        fifo_free(op_queue, operation_delete_func);
    }
}

gboolean
operation_queue_resize(OperationQueue *self, int new_size)
{
    if (new_size == 0) {
        if (self->tile_map) {
            assert(self->dirty_tiles);

            tile_map_free(self->tile_map, TRUE);
            self->tile_map = NULL;
            free(self->dirty_tiles);
            self->dirty_tiles = NULL;
            self->dirty_tiles_n = 0;
        }
        return TRUE;
    } else {
        TileMap *new_tile_map = tile_map_new(new_size, sizeof(Fifo *), free_fifo);
        const int new_map_size = new_size*2*new_size*2;
        TileIndex *new_dirty_tiles = (TileIndex *)malloc(new_map_size*sizeof(TileIndex));

        if (self->tile_map) {
            tile_map_copy_to(self->tile_map, new_tile_map);
            for(int i = 0; i < self->dirty_tiles_n; i++) {
                new_dirty_tiles[i] = self->dirty_tiles[i];
            }

            tile_map_free(self->tile_map, FALSE);
            free(self->dirty_tiles);
        }

        self->tile_map = new_tile_map;
        self->dirty_tiles = new_dirty_tiles;

        return FALSE;
    }
}

OperationQueue *
operation_queue_new()
{
    OperationQueue *self = (OperationQueue *)malloc(sizeof(OperationQueue));

    self->tile_map = NULL;
    self->dirty_tiles_n = 0;
    self->dirty_tiles = NULL;

#ifdef HEAVY_DEBUG
    operation_queue_resize(self, 1);
#else
    operation_queue_resize(self, 10);
#endif

    return self;
}

void
operation_queue_free(OperationQueue *self)
{
    operation_queue_resize(self, 0); // free the tile map data

    free(self);
}

int
tile_equal(TileIndex a, TileIndex b)
{
    return (a.x == b.x && a.y == b.y);
}

size_t
remove_duplicate_tiles(TileIndex *array, size_t length)
{
    if (length < 2) {
        // There cannot be any duplicates
        return length;
    }

    size_t new_length = 1;
    size_t i, j;

    for (i = 1; i < length; i++) {
        for (j = 0; j < new_length; j++) {
            if (tile_equal(array[j], array[i])) {
                break;
            }
        }
        if (j == new_length) {
            array[new_length++] = array[i];
        }
    }
    return new_length;
}

/* Returns all tiles that are have operations queued
 * The consumer that actually does the processing should iterate over this list
 * of tiles, and use operation_queue_pop() to pop all the operations.
 *
 * Concurrency: This function is not thread-safe on the same @self instance. */
int
operation_queue_get_dirty_tiles(OperationQueue *self, TileIndex** tiles_out)
{
    self->dirty_tiles_n = remove_duplicate_tiles(self->dirty_tiles, self->dirty_tiles_n);

    *tiles_out = self->dirty_tiles;
    return self->dirty_tiles_n;
}

/* Clears the list of dirty tiles
 * Consumers should call this after having processed all the tiles.
 *
 * Concurrency: This function is not thread-safe on the same @self instance. */
void
operation_queue_clear_dirty_tiles(OperationQueue *self)
{
    // operation_queue_add will overwrite the invalid tiles as new dirty tiles comes in
    self->dirty_tiles_n = 0;
}

/* Add an operation to the queue for tile @index
 * Note: if an operation affects more than one tile, it must be added once per tile.
 *
 * Concurrency: This function is not thread-safe on the same @self instance. */
void
operation_queue_add(OperationQueue *self, TileIndex index, OperationDataDrawDab *op)
{
    while (!tile_map_contains(self->tile_map, index)) {
#ifdef HEAVY_DEBUG
        operation_queue_resize(self, self->tile_map->size+1);
#else
        operation_queue_resize(self, self->tile_map->size*2);
#endif
    }

    Fifo **queue_pointer = (Fifo **)tile_map_get(self->tile_map, index);
    Fifo *op_queue = *queue_pointer;

    if (op_queue == NULL) {
        // Lazy initialization
        op_queue = fifo_new();

         // Critical section, not thread-safe
        if (!(self->dirty_tiles_n < self->tile_map->size*2*self->tile_map->size*2)) {
            // Prune duplicate tiles that cause us to almost exceed max
            self->dirty_tiles_n = remove_duplicate_tiles(self->dirty_tiles, self->dirty_tiles_n);
        }
        assert(self->dirty_tiles_n < self->tile_map->size*2*self->tile_map->size*2);
        self->dirty_tiles[self->dirty_tiles_n++] = index;
    }

    fifo_push(op_queue, (void *)op);

    *queue_pointer = op_queue;
}

/* Pop an operation off the queue for tile @index
 * The user of this function is reponsible for freeing the result using free()
 *
 * Concurrency: This function is reentrant (and lock-free) on different @index */
OperationDataDrawDab *
operation_queue_pop(OperationQueue *self, TileIndex index)
{
    OperationDataDrawDab *op = NULL;

    if (!tile_map_contains(self->tile_map, index)) {
        return NULL;
    }

    Fifo **queue_pointer = (Fifo **)tile_map_get(self->tile_map, index);
    Fifo *op_queue = *queue_pointer;

    if (!op_queue) {
        return NULL;
    }

    op = (OperationDataDrawDab *)fifo_pop(op_queue);
    if (!op) {
        // Queue empty
        fifo_free(op_queue, operation_delete_func);
        *queue_pointer = NULL;
        return NULL;
    } else {
        assert(op != NULL);
        return op;
    }
}
