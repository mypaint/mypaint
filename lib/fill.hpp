/* This file is part of MyPaint.
 * Copyright (C) 2013 by Andrew Chadwick <a.t.chadwick@gmail.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */

#ifndef __HAVE_FILL
#define __HAVE_FILL

#include "fix15.hpp"


/* Flood fill */


// Pixel access helper

static inline fix15_short_t*
_floodfill_getpixel(PyArrayObject *array,
                    const unsigned int x,
                    const unsigned int y)
{
    const unsigned int xstride = PyArray_STRIDE(array, 1);
    const unsigned int ystride = PyArray_STRIDE(array, 0);
    return (fix15_short_t*)(PyArray_BYTES(array)
                            + (y * ystride)
                            + (x * xstride));
}
                    

// Color match helper

static inline bool
_floodfill_color_match(const fix15_short_t* const pixel,  // premult RGB, and A
                       const fix15_t targ_r, // premult
                       const fix15_t targ_g, // premult
                       const fix15_t targ_b, // premult
                       const fix15_t targ_a)
{
    // The comparison is done using approximately a 7 bit representation
    // for each colour channel.
    const unsigned static int imprecision = 8;

    // Like tile_perceptual_change_strokemap() only (P.{r,g,b,a} * P.alpha) is
    // known for each pixel P, and the same for the target colour. Therefore
    // multiply each component with the alpha of the other colour so that
    // they're in the same base and can be compared.
    const fix15_t a = pixel[3];
    fix15_t r = targ_a * (fix15_t)pixel[0];
    fix15_t g = targ_a * (fix15_t)pixel[1];
    fix15_t b = targ_a * (fix15_t)pixel[2];
    return (   ((targ_r*a) >> imprecision) == (r >> imprecision)
            && ((targ_g*a) >> imprecision) == (g >> imprecision)
            && ((targ_b*a) >> imprecision) == (b >> imprecision)
            && (targ_a >> imprecision) == (a >> imprecision) );
}



typedef struct {
    unsigned int x;
    unsigned int y;
} _floodfill_point;


/*
 * Flood fill one tile from a sequence of seed positions, and return overflows
 *
 * Returns four lists of overflow coordinates denoting on which pixels of the
 * next tile to the N, E, S, and W the fill has overflowed. These coordinates
 * can be fed back unto tile_flood_fill() for the tile identified.
 */

PyObject *
tile_flood_fill (PyObject *tile, /* HxWx4 array of uint16 */
                 PyObject *seeds, /* List of 2-tuples */
                 int targ_r, int targ_g, int targ_b, int targ_a, //premult
                 double fill_r, double fill_g, double fill_b,
                 int min_x, int min_y, int max_x, int max_y)
{
    // Fill colour args are floats [0.0 .. 1.0], non-premultiplied by alpha.
    // The targ_ colour components are 15-bit scaled ints in the range
    // [0 .. 1<<15], and are premultiplied by targ_a which has the same range.
    fix15_short_t targ_r15 = fix15_short_clamp(targ_r);
    fix15_short_t targ_g15 = fix15_short_clamp(targ_g);
    fix15_short_t targ_b15 = fix15_short_clamp(targ_b);
    fix15_short_t targ_a15 = fix15_short_clamp(targ_a);
    fix15_short_t fill_r15 = fix15_short_clamp(fill_r * fix15_one);
    fix15_short_t fill_g15 = fix15_short_clamp(fill_g * fix15_one);
    fix15_short_t fill_b15 = fix15_short_clamp(fill_b * fix15_one);
    PyArrayObject *tile_arr = ((PyArrayObject *)tile);
    // Dimensions are [y][x][component]
#ifdef HEAVY_DEBUG
    assert(PyArray_Check(tile));
    assert(PyArray_DIM(tile_arr, 0) == MYPAINT_TILE_SIZE);
    assert(PyArray_DIM(tile_arr, 1) == MYPAINT_TILE_SIZE);
    assert(PyArray_DIM(tile_arr, 2) == 4);
    assert(PyArray_TYPE(tile_arr) == NPY_UINT16);
    assert(PyArray_ISCARRAY(tile_arr));
    static const unsigned int xstride = PyArray_STRIDES(tile_arr)[1];
    assert(xstride == 4*sizeof(uint16_t));
    static const unsigned int ystride = PyArray_STRIDES(tile_arr)[0];
    assert(ystride == MYPAINT_TILE_SIZE * xstride);
    assert(PyArray_STRIDES(tile_arr)[2] == sizeof(uint16_t));
    assert(PySequence_Check(seeds));
    int num_compared = 1;
#endif
    if (min_x < 0) min_x = 0;
    if (min_y < 0) min_y = 0;
    if (max_x >= MYPAINT_TILE_SIZE) max_x = MYPAINT_TILE_SIZE;
    if (max_y >= MYPAINT_TILE_SIZE) max_y = MYPAINT_TILE_SIZE;
    if (min_x > max_x || min_y > max_y) {
        return Py_BuildValue("[]");
    }

    // Populate a working queue with seeds
    int x, y;
    GQueue *queue = g_queue_new();   /* Of tuples, to be exhausted */
    for (int i=0; i<PySequence_Size(seeds); ++i) {
        PyObject *seed_tup = PySequence_GetItem(seeds, i);
        x = (int) PyInt_AsLong(PySequence_GetItem(seed_tup, 0));
        y = (int) PyInt_AsLong(PySequence_GetItem(seed_tup, 1));
        // Skip seed point if we've already been here
        const fix15_short_t *pixel = _floodfill_getpixel(tile_arr, x, y);
        if ( (pixel[0] == fill_r15) &&
             (pixel[1] == fill_g15) &&
             (pixel[2] == fill_b15) &&
             (pixel[3] == fix15_one) ) {
            continue;
        }
        // Enqueue the seed point if it matches the target colour
        if (_floodfill_color_match(pixel, targ_r15, targ_g15, targ_b15,
                                          targ_a15))
        {
            _floodfill_point *seed_pt = (_floodfill_point*)
                                          malloc(sizeof(_floodfill_point));
            seed_pt->x = x;
            seed_pt->y = y;
            g_queue_push_tail(queue, seed_pt);
        }
    }

    PyObject *result_n = PyList_New(0);
    PyObject *result_e = PyList_New(0);
    PyObject *result_s = PyList_New(0);
    PyObject *result_w = PyList_New(0);

    while (! g_queue_is_empty(queue)) {
        _floodfill_point *pos = (_floodfill_point*) g_queue_pop_head(queue);
        int x0 = pos->x;
        int y = pos->y;
        free(pos);
#ifdef HEAVY_DEBUG
        ++num_compared;
#endif
        // Find easternmost and westernmost points of the same colour
        // Westwards loop includes (x,y), eastwards ignores it.
        static const int x_delta[] = {-1, 1};
        static const int x_offset[] = {0, 1};
        for (int i=0; i<2; ++i)
        {
            for ( int x = x0 + x_offset[i] ;
                  x >= min_x && x < max_x ;
                  x += x_delta[i] )
            {
                // Halt expansion if we've already filled this pixel
                fix15_short_t *pixel = _floodfill_getpixel(tile_arr, x, y);
                if ( (pixel[0] == fill_r15) &&
                     (pixel[1] == fill_g15) &&
                     (pixel[2] == fill_b15) &&
                     (pixel[3] == fix15_one) ) {
                    break;
                }
                // Halt expansion if the pixel doesn't match the target colour.
                if (! _floodfill_color_match(pixel, targ_r15, targ_g15, targ_b15,
                                                    targ_a15))
                {
                    break;
                }
                // Also halt if we're outside the bbox range
                if (x < min_x || y < min_y || x >= max_x || y >= max_y) {
                    break;
                }
                // Fill this pixel, and continue iterating in this direction.
                // In addition, enqueue the pixels above and below that
                // matched.
                pixel[0] = fix15_short_clamp(fill_r * fix15_one);
                pixel[1] = fix15_short_clamp(fill_g * fix15_one);
                pixel[2] = fix15_short_clamp(fill_b * fix15_one);
                pixel[3] = fix15_one;
                if (y > 0) {
                    // Enqueue the pixel to the north
                    _floodfill_point *p = (_floodfill_point *)
                                            malloc(sizeof(_floodfill_point));
                    p->x = x;
                    p->y = y-1;
                    g_queue_push_tail(queue, p);
                }
                else {
                    // Overflow onto the tile to the North
                    PyObject *s = Py_BuildValue("ii", x, MYPAINT_TILE_SIZE-1);
                    PyList_Append(result_n, s);
                }
                if (y < MYPAINT_TILE_SIZE - 1) {
                    // Enqueue the pixel to the South
                    _floodfill_point *p = (_floodfill_point *)
                                            malloc(sizeof(_floodfill_point));
                    p->x = x;
                    p->y = y+1;
                    g_queue_push_tail(queue, p);
                }
                else {
                    // Overflow onto the tile to the South
                    PyObject *s = Py_BuildValue("ii", x, 0);
                    PyList_Append(result_s, s);
                }
                // If the fill is now at the west or east extreme, we have
                // overflowed.  Seed West and East tiles.
                if (x == 0) {
                    PyObject *s = Py_BuildValue("ii", MYPAINT_TILE_SIZE-1, y);
                    PyList_Append(result_w, s);
                }
                else if (x == MYPAINT_TILE_SIZE-1) {
                    PyObject *s = Py_BuildValue("ii", 0, y);
                    PyList_Append(result_e, s);
                }
            }
        }
    }
    
    g_queue_free(queue);
    return Py_BuildValue("[OOOO]", result_n, result_e, result_s, result_w);
}




#endif //__HAVE_FILL
