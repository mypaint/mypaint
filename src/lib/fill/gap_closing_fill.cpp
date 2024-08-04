/* This file is part of MyPaint.
 * Copyright (C) 2018 by the MyPaint Development Team.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */

#include "floodfill.hpp"

#include <cmath>
#include <vector>

// Largest squared gap distance - represents infinity for distances
#define MAX_GAP (2 * N * N)

// Check if an outgoing edge is fully seeded with infinite distances
static bool
all_max_dist(chan_t seeds[N])
{
    for (int i = 0; i < N + 10; ++i) {
        if (seeds[i] != MAX_GAP) return false;
    }
    return true;
}

// Gap closing requires each seed to keep track of the maximum
// detected distance it encountered on its path, hence ranges
// are not used here.
static inline PyObject*
simple_seeds(chan_t seeds[N], edge e)
{
    // For large fills (perhaps accidentally so, when leaking through)
    // these shortcuts are used to potentially skip large areas of
    // empty tiles without detected gaps
    if (all_max_dist(seeds)) {
        // This value is queued for a neighbouring tile, so the direction
        // is inverted to be interpreted as "coming from this direction"
        edge inverted = edge((e + 2) % 4);
        return Py_BuildValue("(i)", inverted);
    }

    // Fall back to creating seed tuples in a list.
    PyObject* list = PyList_New(0);

    for (int i = 0; i < N; ++i) {
        chan_t d = seeds[i];
        if (d == 0) continue;

        PyObject* seed;
        switch (e) {
        case edges::north:
            seed = Py_BuildValue("iii", i, N - 1, d);
            break;
        case edges::east:
            seed = Py_BuildValue("iii", 0, i, d);
            break;
        case edges::south:
            seed = Py_BuildValue("iii", i, 0, d);
            break;
        case edges::west:
            seed = Py_BuildValue("iii", N - 1, i, d);
            break;
        default:
            throw;
        }
        PyList_Append(list, seed);
        Py_DECREF(seed);
#ifdef HEAVY_DEBUG
        assert(seed->ob_refcnt == 1);
#endif
    }
    return list;
}

/*
    Gap closing queue item.
    The is_seed state is only set for incoming seeds,
    in order to not create redundant outgoing seeds.
*/
struct gc_coord {
    gc_coord() {}
    gc_coord(int x, int y, chan_t d) : x(x), y(y), distance(d), is_seed(false)
    {
    }
    int x;
    int y;
    chan_t distance;
    bool is_seed;
};

#define GC_DIFF_LIMIT 2.0
#define GC_TRACK_MIN true

GapClosingFiller::GapClosingFiller(int max_dist, bool track_seep)
    : max_distance(max_dist), track_seep(track_seep)
{
}

// Queues a single gc_coord or marks it as an outgoing seed by
// setting its distance in the relevant out-seed array.
static void
queue_gc_seeds(
    std::queue<gc_coord>& queue, gc_coord& c, chan_t curr_dist,
    chan_t north[], chan_t east[], chan_t south[], chan_t west[])
{
    int x = c.x;
    int y = c.y;
    bool not_seed = !c.is_seed;

    if (y > 0)
        queue.push(gc_coord(x, y - 1, curr_dist));
    else if (not_seed)
        north[x] = curr_dist;

    if (y < N - 1)
        queue.push(gc_coord(x, y + 1, curr_dist));
    else if (not_seed)
        south[x] = curr_dist;

    if (x > 0)
        queue.push(gc_coord(x - 1, y, curr_dist));
    else if (not_seed)
        west[y] = curr_dist;

    if (x < N - 1)
        queue.push(gc_coord(x + 1, y, curr_dist));
    else if (not_seed)
        east[y] = curr_dist;
}

static void
populate_gc_queue(std::queue<gc_coord>& queue, PyObject* seeds)
{
    if (PyTuple_CheckExact(seeds)) {
        edge origin;
        PyArg_ParseTuple(seeds, "i", &origin);

        int x_base = (origin == edges::east) * (N - 1);
        int y_base = (origin == edges::south) * (N - 1);
        int x_offs = (origin + 1) % 2;
        int y_offs = origin % 2;
        for (int i = 0; i < N; ++i) {
            const int x = x_base + (x_offs * i);
            const int y = y_base + (y_offs * i);
            gc_coord seed = gc_coord(x, y, MAX_GAP);
            seed.is_seed = true;
            queue.push(seed);
        }
        return;
    }

    // Populate the queue
    for (int i = 0; i < PySequence_Size(seeds); ++i) {

        gc_coord seed_pt;
        PyObject* tuple = PySequence_GetItem(seeds, i);
        PyArg_ParseTuple(
            tuple, "iii", &(seed_pt.x), &(seed_pt.y), &seed_pt.distance);
        seed_pt.is_seed = true;
        Py_DECREF(tuple);
#ifdef HEAVY_DEBUG
        assert(tuple->ob_refcnt == 1);
#endif
        queue.push(seed_pt);
    }
}

PyObject*
GapClosingFiller::fill(
    PyObject* alphas_arr, PyObject* dists_arr, PyObject* dst_arr,
    PyObject* seeds, int min_x, int min_y, int max_x, int max_y)
{

    if (min_x > max_x || min_y > max_y) return Py_BuildValue("[()()()()()0]");
    if (min_x < 0) min_x = 0;
    if (min_y < 0) min_y = 0;
    if (max_x > (N - 1)) max_x = (N - 1);
    if (max_y > (N - 1)) max_y = (N - 1);

    PixelBuffer<chan_t> alphas(alphas_arr);
    PixelBuffer<chan_t> distances(dists_arr);
    PixelBuffer<chan_t> dst(dst_arr);

    std::queue<gc_coord> queue;
    populate_gc_queue(queue, seeds);

    std::vector<gc_coord> fill_edges;

    chan_t north[N] = {
        0,
    };
    chan_t east[N] = {
        0,
    };
    chan_t south[N] = {
        0,
    };
    chan_t west[N] = {
        0,
    };

    int pixels_filled = 0;

    while (!queue.empty()) {
        gc_coord c = queue.front();
        int x = c.x;
        int y = c.y;
        queue.pop();

        if (x < min_x || x > max_x || y < min_y || y > max_y) continue;

        chan_t alpha = alphas(x, y);
        if (dst(x, y) != 0 || (alpha <= 0)) continue;

        const chan_t prev_dist = c.distance;
        chan_t curr_dist = distances(x, y);

        if (prev_dist != curr_dist) {

            // Crude fill-in of isolated unassigned pixels.
            // Blurring the distance tiles would be an expensive alternative.
            if (curr_dist == MAX_GAP) {
                if (track_seep) {

                    if (x > 0 && x < N - 1) {

                        if ((dst(x - 1, y) != 0 || alphas(x - 1, y) == 0) &&
                            (dst(x + 1, y) != 0 || alphas(x + 1, y) == 0)) {

                            dst(x, y) = alpha;
                            continue;
                        }
                    }

                    if (y > 0 && y < N - 1) {

                        if ((dst(x, y - 1) != 0 || alphas(x, y - 1) == 0) &&
                            (dst(x, y + 1) != 0 || alphas(x, y + 1) == 0)) {

                            dst(x, y) = alpha;
                            continue;
                        }
                    }

                    fill_edges.push_back(gc_coord(x, y, curr_dist));
                }
                continue;
            } else if (
                prev_dist < curr_dist &&
                sqrtf(curr_dist) - sqrtf(prev_dist) > GC_DIFF_LIMIT) {
                if (track_seep) {
                    fill_edges.push_back(gc_coord(x, y, curr_dist));
                }
                continue;
            }

            if (GC_TRACK_MIN && prev_dist < curr_dist) curr_dist = prev_dist;
        }

        pixels_filled++;
        dst(x, y) = alpha;

        // Queue adjacent pixels
        queue_gc_seeds(queue, c, curr_dist, north, east, south, west);
    }

    PyObject* f_edge_list = PyList_New(0);

    for (std::vector<gc_coord>::iterator i = fill_edges.begin();
         i != fill_edges.end(); ++i) {
        if (dst(i->x, i->y) == 0) {
            PyObject* tuple = Py_BuildValue("iii", i->x, i->y, i->distance);
            PyList_Append(f_edge_list, tuple);
            Py_DECREF(tuple);
#ifdef HEAVY_DEBUG
            assert(tuple->ob_refcnt == 1);
#endif
        }
    }

    return Py_BuildValue(
        "[NNNNNi]", simple_seeds(north, edges::north),
        simple_seeds(east, edges::east), simple_seeds(south, edges::south),
        simple_seeds(west, edges::west), f_edge_list, pixels_filled);
}

// Based on coordinates of places where the initial fill stopped due
// to leaving an area with tracked distances, move back into the fill,
// erasing it until the tracked distances grow at a certain rate.
PyObject*
GapClosingFiller::unseep(
    PyObject* dists_arr, PyObject* dst_arr, PyObject* seeds, bool initial)
{
    PixelBuffer<chan_t> distances(dists_arr);
    PixelBuffer<chan_t> dst(dst_arr);

    std::queue<gc_coord> queue;
    // Populate the queue
    for (int i = 0; i < PySequence_Size(seeds); ++i) {

        gc_coord seed_pt;
        PyObject* tuple = PySequence_GetItem(seeds, i);
        PyArg_ParseTuple(
            tuple, "iii", &(seed_pt.x), &(seed_pt.y), &seed_pt.distance);
        seed_pt.is_seed = true;
        Py_DECREF(tuple);
#ifdef HEAVY_DEBUG
        assert(tuple->ob_refcnt == 1);
#endif

        // Don't queue initial track_seep seeds that were filled from another
        // direction. Initial seeds are created during the fill process,
        // and the rest are created as part of the unseep process.
        if (initial ^ (dst(seed_pt.x, seed_pt.y) != 0)) {
            dst(seed_pt.x, seed_pt.y) = fix15_one;
            queue.push(seed_pt);
        }
    }

    chan_t north[N] = {
        0,
    };
    chan_t east[N] = {
        0,
    };
    chan_t south[N] = {
        0,
    };
    chan_t west[N] = {
        0,
    };

    int pixels_erased = 0;

    // Erase loop
    while (!queue.empty()) {
        gc_coord c = queue.front();
        int x = c.x;
        int y = c.y;
        bool not_seed = !c.is_seed;
        queue.pop();

        const chan_t prev_dist = c.distance;
        chan_t curr_dist = distances(x, y);

        if (dst(x, y) == 0) continue;

        pixels_erased++;
        dst(x, y) = 0;

        if (curr_dist == MAX_GAP && not_seed) continue;

        if (prev_dist != curr_dist) {

            if (curr_dist == MAX_GAP ||
                (prev_dist < curr_dist &&
                 // The marked distances are actually squared, hence the
                 // square root is taken before checking the delta
                 sqrtf((float)curr_dist) - sqrtf((float)prev_dist) > 1)) {
                continue;
            }

            if (prev_dist < curr_dist) // Always track minimum when unseeping
                curr_dist = prev_dist;
        }
        queue_gc_seeds(queue, c, curr_dist, north, east, south, west);
    }
    return Py_BuildValue(
        "[NNNNi]", simple_seeds(north, edges::north),
        simple_seeds(east, edges::east), simple_seeds(south, edges::south),
        simple_seeds(west, edges::west), pixels_erased);
}
