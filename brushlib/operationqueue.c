
#include <malloc.h>
#include <assert.h>

#include "operationqueue.h"

struct _OperationQueue {
    OperationDataDrawDab **tile_map;
    TileIndex origin;
    int tile_map_dim_size;
};


OperationQueue *
operation_queue_new()
{
    OperationQueue *self = (OperationQueue *)malloc(sizeof(OperationQueue));

    self->tile_map_dim_size = 20;
    self->origin.x = self->tile_map_dim_size;
    self->origin.y = self->tile_map_dim_size;
    const int tile_map_size = 2*self->tile_map_dim_size*2*self->tile_map_dim_size;
    self->tile_map = (OperationDataDrawDab **)malloc(tile_map_size*sizeof(OperationDataDrawDab *));
    for(int i = 0; i < tile_map_size; i++) {
        self->tile_map[i] = NULL;
    }

    return self;
}

void
operation_queue_free(OperationQueue *self)
{
    free(self);
}


OperationDataDrawDab **
tile_map_get(OperationQueue *self, TileIndex index)
{
    const int offset = ((self->origin.y + index.y) * self->tile_map_dim_size*2) + self->origin.x + index.x;
    assert(offset < 2*self->tile_map_dim_size*2*self->tile_map_dim_size);
    assert(offset >= 0);
    return self->tile_map + offset;
}

typedef void (*TileMapForeachFunction)(OperationDataDrawDab *op, TileIndex index, void *user_data);

void
tile_map_foreach(OperationQueue *self, TileMapForeachFunction function, void *user_data)
{
    for(int i = 0; i < self->tile_map_dim_size*2; i++) {
        for(int j = 0; j < self->tile_map_dim_size*2; j++) {

            TileIndex index;
            index.x = j - self->origin.x;
            index.y = i - self->origin.y;

            OperationDataDrawDab *op = *tile_map_get(self, index);
            function(op, index, user_data);
        }
    }
}

void
count_tiles_with_ops(OperationDataDrawDab *op, TileIndex index, void *user_data)
{
    int *number_of_tiles = (int *)user_data;
    if (op) {
        (*number_of_tiles)++;
    }
}

typedef struct {
    TileIndex *tiles;
    int tile_no;
} CollectTilesState;

void
collect_tiles_with_ops(OperationDataDrawDab *op, TileIndex index, void *user_data)
{
    CollectTilesState *state = (CollectTilesState *)user_data;
    if (op) {
        state->tiles[state->tile_no++] = index;
    }
}

/* Returns all tiles that are have operations queued */
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

void
operation_queue_add(OperationQueue *self, TileIndex index, OperationDataDrawDab *op)
{
    assert(index.x < self->tile_map_dim_size);
    assert(index.y < self->tile_map_dim_size);

    OperationDataDrawDab **op_pointer = tile_map_get(self, index);
    assert(*op_pointer == NULL);
    *op_pointer = op;
}

OperationDataDrawDab *
operation_queue_pop(OperationQueue *self, TileIndex index)
{
    assert(index.x < self->tile_map_dim_size);
    assert(index.y < self->tile_map_dim_size);

    OperationDataDrawDab **op_pointer = tile_map_get(self, index);
    OperationDataDrawDab *op = *op_pointer;
    //assert(op != NULL);
    *op_pointer = NULL;
    return op;
}
