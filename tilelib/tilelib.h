#ifndef __TILELIB_H__
#define __TILELIB_H__

#define TILE_SIZE 64

void tile_draw_dab (PyObject * tiled_surface, 
                    float x, float y, 
                    float radius, 
                    float color_r, float color_g, float color_b,
                    float opaque, float hardness
                    );

/*
void tile_get_color (PyObject * tiled_surface, 
...
                     );
*/

#endif
