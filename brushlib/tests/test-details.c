
#include <stdlib.h>
#include <stdio.h>

#include "mypaint-tiled-surface.h"
#include "tiled-surface-private.h"
#include "mypaint-benchmark.h"

// TODO: test
// Tile requests


int main(int argc, char *argv[])
{
    // 
    const int x = 0;
    const int y = 0;
    const float radius = MYPAINT_TILE_SIZE/2;
    const float hardness = 1.0;
    const float angle = 0.0;
    const float aspect_ratio = 1.0;

    const int iterations = 1000000;

    uint16_t buffer[MYPAINT_TILE_SIZE*MYPAINT_TILE_SIZE+2*MYPAINT_TILE_SIZE];
    mypaint_benchmark_start("render_dab_mask");
    for (int i=0; i < iterations; i++) {
        render_dab_mask(buffer, x, y, radius, hardness, aspect_ratio, angle);
    }
    const int duration = mypaint_benchmark_end();
    printf("render_dab_mask: %d ms\n", duration);
}
