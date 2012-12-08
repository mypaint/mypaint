
#include <malloc.h>
#include <assert.h>

#include <mypaint-glib-compat.h>
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

/* For use with queue_delete */
void
operation_delete_func(void *user_data) {
    if (user_data) {
        free(user_data);
    }
}

Fifo **
tile_map_new(int new_map_size)
{
    Fifo **new_tile_map = (Fifo **)malloc(new_map_size*sizeof(Fifo *));
    for(int i = 0; i < new_map_size; i++) {
        new_tile_map[i] = NULL;
    }
    return new_tile_map;
}

void
tile_map_free(OperationQueue *self)
{
    const int tile_map_size = 2*self->tile_map_dim_size*2*self->tile_map_dim_size;
    for(int i = 0; i < tile_map_size; i++) {
        Fifo *op_queue = self->tile_map[i];
        if (op_queue) {
            fifo_free(op_queue, operation_delete_func);
        }
    }
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

void
tile_map_copy(OperationQueue *self, Fifo **new_tile_map, int new_size)
{
    const int old_map_size = 2*self->tile_map_dim_size*2*self->tile_map_dim_size;

    // FIXME: need to take origin into account to be able to copy to the correct location
    for(int i = 0; i < old_map_size; i++) {
        new_tile_map[i] = self->tile_map[i];
    }
}


gboolean
operation_queue_resize(OperationQueue *self, int new_size)
{
    if (new_size == 0) {
        if (self->tile_map_dim_size != 0) {
            assert(self->tile_map);
            assert(self->dirty_tiles);

            tile_map_free(self);
            self->tile_map_dim_size = 0;

            self->dirty_tiles_n = 0;
            free(self->dirty_tiles);
        }
        return TRUE;
    } else {
        const int new_map_size = 2*new_size*2*new_size;
        Fifo **new_tile_map = tile_map_new(new_map_size);
        TileIndex *new_dirty_tiles = (TileIndex *)malloc(new_map_size*sizeof(TileIndex));

        // Copy old values over
        tile_map_copy(self, new_tile_map, new_size);

        for(int i = 0; i < self->dirty_tiles_n; i++) {
            new_dirty_tiles[i] = self->dirty_tiles[i];
        }

        // Free old values
        tile_map_free(self);
        free(self->dirty_tiles);

        // Set new values
        self->tile_map_dim_size = new_size;
        self->origin.x = self->tile_map_dim_size;
        self->origin.y = self->tile_map_dim_size;
        self->tile_map = new_tile_map;
        self->dirty_tiles = new_dirty_tiles;

        return FALSE;
    }
}

OperationQueue *
operation_queue_new()
{
    OperationQueue *self = (OperationQueue *)malloc(sizeof(OperationQueue));

    self->tile_map_dim_size = 0;
    self->tile_map = NULL;
    self->dirty_tiles_n = 0;
    self->dirty_tiles = NULL;
    self->origin.x = 0;
    self->origin.y = 0;

    operation_queue_resize(self, 10);

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

static gboolean
index_is_outside_map(OperationQueue *self, TileIndex index)
{
    return (abs(index.x) >= self->tile_map_dim_size ||
            abs(index.y) >= self->tile_map_dim_size);
}

/* Add an operation to the queue for tile @index
 * Note: if an operation affects more than one tile, it must be added once per tile.
 *
 * Concurrency: This function is not thread-safe on the same @self instance. */
void
operation_queue_add(OperationQueue *self, TileIndex index, OperationDataDrawDab *op)
{
    while (index_is_outside_map(self, index)) {
        operation_queue_resize(self, self->tile_map_dim_size*2);
    }

    // FIXME: valgrind says this sometimes causes a write outsize the end of the array
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

    if (index_is_outside_map(self, index)) {
        return NULL;
    }

    // FIXME: valgrind says this sometimes causes a write outsize the end of the array
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
