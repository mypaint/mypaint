
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
};

/* FIXME: Remove hardcoding of tile map size. When an index outside the current map is requested, the map needs to grown dynamically */
/* FIXME: actually queue, and handle more than one operation per tile index. */

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

typedef void (*TileMapForeachFunction)(Fifo *op_queue, TileIndex index, void *user_data);

void
tile_map_foreach(OperationQueue *self, TileMapForeachFunction function, void *user_data)
{
    for(int i = 0; i < self->tile_map_dim_size*2; i++) {
        for(int j = 0; j < self->tile_map_dim_size*2; j++) {

            TileIndex index;
            index.x = j - self->origin.x;
            index.y = i - self->origin.y;

            Fifo *op_queue = *tile_map_get(self, index);
            function(op_queue, index, user_data);
        }
    }
}

void
count_tiles_with_ops(Fifo *op_queue, TileIndex index, void *user_data)
{
    int *number_of_tiles = (int *)user_data;
    if (op_queue) {
        (*number_of_tiles)++;
    }
}

typedef struct {
    TileIndex *tiles;
    int tile_no;
} CollectTilesState;

void
collect_tiles_with_ops(Fifo *op_queue, TileIndex index, void *user_data)
{
    CollectTilesState *state = (CollectTilesState *)user_data;
    if (op_queue) {
        state->tiles[state->tile_no++] = index;
    }
}

/* Returns all tiles that are have operations queued
 * The consumer that actually does the processing should iterate over this list
 * of tiles, and use operation_queue_pop() to pop all the operations. */
int
operation_queue_get_tiles(OperationQueue *self, TileIndex** tiles_out)
{
    int number_of_tiles = 0;
    tile_map_foreach(self, count_tiles_with_ops, &number_of_tiles);

    TileIndex *tiles = (TileIndex *)malloc(number_of_tiles*sizeof(TileIndex));
    CollectTilesState temp_state = {tiles, 0};
    tile_map_foreach(self, collect_tiles_with_ops, &temp_state);

    assert(temp_state.tile_no == number_of_tiles);

    *tiles_out = tiles;
    return number_of_tiles;
}

/* Add an operation to the queue for tile @index
 * Note: if an operation affects more than one tile, it must be added once per tile.
 *
 * Concurrency: This function is reentrant (and lock-free) on different @index */
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
