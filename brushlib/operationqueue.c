
#include <malloc.h>
#include <assert.h>

#include "operationqueue.h"
#include "fifo.h"

static const int OP_QUEUE_SIZE = 10;

typedef struct fifo Fifo;

struct _OperationQueue {
    Fifo **tile_map;
    TileIndex origin;
    int tile_map_dim_size;

    TileIndex *dirty_tiles;
    int dirty_tiles_n;
};

/* FIXME: Remove hardcoding of tile map size.
 * When an index outside the current map is requested,
 * the map needs to grown dynamically. */

OperationQueue *
operation_queue_new()
{
    OperationQueue *self = (OperationQueue *)malloc(sizeof(OperationQueue));

    self->tile_map_dim_size = 100;
    self->origin.x = self->tile_map_dim_size;
    self->origin.y = self->tile_map_dim_size;
    const int tile_map_size = 2*self->tile_map_dim_size*2*self->tile_map_dim_size;
    self->tile_map = (Fifo **)malloc(tile_map_size*sizeof(Fifo *));
    for(int i = 0; i < tile_map_size; i++) {
        self->tile_map[i] = NULL;
    }

    self->dirty_tiles = (TileIndex *)malloc(tile_map_size*sizeof(TileIndex));
    self->dirty_tiles_n = 0;

    return self;
}

/* For use with queue_delete */
void
operation_delete_func(void *user_data) {
    if (user_data) {
        free(user_data);
    }
}


void
operation_queue_free(OperationQueue *self)
{
    const int tile_map_size = 2*self->tile_map_dim_size*2*self->tile_map_dim_size;
    for(int i = 0; i < tile_map_size; i++) {
        Fifo *op_queue = self->tile_map[i];
        if (op_queue) {
            // fifo_free(op_queue, operation_delete_func);
        }
    }
    free(self->tile_map);

    free(self->dirty_tiles);

    free(self);
}

/* Get the data in the tile map for a given tile @index.
 * Must be reentrant and lock-free on different @index */
Fifo **
tile_map_get(OperationQueue *self, TileIndex index)
{
    const int offset = ((self->origin.y + index.y) * self->tile_map_dim_size*2) + self->origin.x + index.x;
    assert(offset < 2*self->tile_map_dim_size*2*self->tile_map_dim_size);
    assert(offset >= 0);
    return self->tile_map + offset;
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
    assert(index.x < self->tile_map_dim_size);
    assert(index.y < self->tile_map_dim_size);

    Fifo **queue_pointer = tile_map_get(self, index);
    Fifo *op_queue = *queue_pointer;

    if (op_queue == NULL) {
        // Lazy initialization
        op_queue = fifo_new();
        self->dirty_tiles[self->dirty_tiles_n++] = index; // Critical section, not thread-safe
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
    assert(index.x < self->tile_map_dim_size);
    assert(index.y < self->tile_map_dim_size);

    Fifo **queue_pointer = tile_map_get(self, index);
    Fifo *op_queue = *queue_pointer;

    if (!op_queue) {
        return NULL;
    }

    op = fifo_pop(op_queue);
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
