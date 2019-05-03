/* This file is part of MyPaint.
 * Copyright (C) 2018 by the MyPaint Development Team.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */
#include "floodfill.hpp"

#include <queue>
#include <stdio.h>
#include <string.h>
#include <cmath>

/*
  Returns a list [(start, end)] of segments corresponding
  to contiguous occurences of "true" in the given boolean array.
  This is the representation that is passed between the Python
  and C++ parts of the fill algorithm, representing, along with a
  direction marker, sequences of fill seeds to be handled.
*/
static inline PyObject*
to_seeds(const bool edge[N])
{
    PyObject* seg_list = PyList_New(0);

    bool contiguous = false;
    int seg_start = 0;
    int seg_end = 0;

    for (int c = 0; c < N; ++c) {
        if (edge[c]) {
            if (!contiguous) { // a new segment begins
                seg_start = c;
                seg_end = c;
                contiguous = true;
            } else { // segment continues
                seg_end++;
            }
        } else if (contiguous) { // segment ends
            PyObject* segment = Py_BuildValue("ii", seg_start, seg_end);
            PyList_Append(seg_list, segment);
            Py_DECREF(segment);
#ifdef HEAVY_DEBUG
            assert(segment->ob_refcnt == 1);
#endif
            contiguous = false;
        }
    }
    if (contiguous) {
        PyObject* segment = Py_BuildValue("ii", seg_start, seg_end);
        PyList_Append(seg_list, segment);
        Py_DECREF(segment);
#ifdef HEAVY_DEBUG
        assert(segment->ob_refcnt == 1);
#endif
    }

    return seg_list;
}

static inline chan_t
clamped_div(chan_t a, chan_t b)
{
    return fix15_short_clamp(fix15_div(fix15_short_clamp(a), b));
}

static inline rgba
straightened(int targ_r, int targ_g, int targ_b, int targ_a)
{
    if (targ_a <= 0)
        return rgba((chan_t)0, 0, 0, 0);
    else
        return rgba(
            clamped_div(targ_r, targ_a), clamped_div(targ_g, targ_a),
            clamped_div(targ_b, targ_a), targ_a);
}

Filler::Filler(int targ_r, int targ_g, int targ_b, int targ_a, double tol)
    : targ(straightened(targ_r, targ_g, targ_b, targ_a)),
      targ_premult(rgba((chan_t)targ_r, targ_g, targ_b, targ_a)),
      tolerance((fix15_t)(MIN(1.0, MAX(0.0, tol)) * fix15_one))
{
}

/*
  Evaluates the source pixel based on the target color and
  tolerance value, returning the alpha of the fill as it
  should be set for the corresponding destination pixel.

  A non-zero result indicates that the destination pixel
  should be filled.
*/
chan_t
Filler::pixel_fill_alpha(const rgba& px)
{
    fix15_t dist;

    if ((targ.alpha | px.alpha) == 0)
        return fix15_one;
    else if (tolerance == 0) {
        return fix15_one * (targ_premult == px);
    }

    if (targ.alpha == 0)
        dist = px.alpha;
    else
        dist = targ.max_diff(straightened(px.red, px.green, px.blue, px.alpha));

    // Compare with adjustable tolerance of mismatches.
    static const fix15_t onepointfive = fix15_one + fix15_halve(fix15_one);
    dist = fix15_div(dist, tolerance);
    if (dist > onepointfive) { // aa < 0, but avoid underflow
        return 0;
    } else {
        fix15_t aa = onepointfive - dist;
        if (aa < fix15_halve(fix15_one))
            return fix15_short_clamp(fix15_double(aa));
        else
            return fix15_one;
    }
}

/*
  Helper function for the fill algorithm, enqueues the source pixel (at (x,y))
  if the corresponding destination pixel is not already filled, the source
  pixel color is within the fill tolerance and the pixel is not part of
  a contiguous sequence (on the same row) already represented in the queue.

  Return value indicates:
  "sequence is no longer contigous, must enqueue next eligible pixel"
*/
bool
Filler::check_enqueue(
    const int x, const int y, bool check, const rgba& src_pixel,
    const chan_t& dst_pixel)
{
    if (dst_pixel != 0) return true;
    bool match = pixel_fill_alpha(src_pixel) > 0;
    if (match && check) {
        queue.push(coord(x, y));
        return false;
    }
    return !match;
}

/*
  If the input tile is:

  Not uniform - returns Py_None
  If uniform, returns Tolerance value as a Py_Int
*/
PyObject*
Filler::tile_uniformity(bool empty_tile, PyObject* src_arr)
{
    if (empty_tile) {
        chan_t alpha = pixel_fill_alpha(rgba((chan_t)0, 0, 0, 0));
        return Py_BuildValue("i", alpha);
    }

    PixelBuffer<rgba> src = PixelBuffer<rgba>(src_arr);

    if (src.is_uniform()) {
        chan_t alpha = pixel_fill_alpha(src(0, 0));
        return Py_BuildValue("i", alpha);
    }

    Py_INCREF(Py_None);
    return Py_None;
}

/*
  Four-way fill algorithm using segments to represent
  sequences of input and output seeds.

  The src and dst buffers are wrapped python array references
  enabling abstracted pointer arithmetic.

  The seed_origin parameter denotes from which edge the input
  seeds are coming, and is used to initiate and finalize the
  input and output seeds respectively.
*/
PyObject*
Filler::fill(
    PyObject* src_o, PyObject* dst_o, PyObject* seeds, edge seed_origin,
    int min_x, int min_y, int max_x, int max_y)
{
    if (min_x > max_x || min_y > max_y) return Py_BuildValue("[()()()()]");
    if (min_x < 0) min_x = 0;
    if (min_y < 0) min_y = 0;
    if (max_x > (N - 1)) max_x = (N - 1);
    if (max_y > (N - 1)) max_y = (N - 1);

    PixelBuffer<rgba> src(src_o);
    PixelBuffer<chan_t> dst(dst_o);

#ifdef HEAVY_DEBUG
    assert(PySequence_Check(seeds));
#endif

    // Store input seed positions to filter them out
    // prior to constructing the output seed segment lists
    bool input_seeds[N] = {
        0,
    };

    if (PyTuple_CheckExact(seeds)) { // the very first seed
        coord seed_pt;
        PyArg_ParseTuple(seeds, "ii", &(seed_pt.x), &(seed_pt.y));
        queue.push(seed_pt);
    } else {
#ifdef HEAVY_DEBUG
        assert(PySequence_Check(seeds));
#endif

        // The tile edge and direction determined by where the seed segments
        // originates, left-to-right or top-to-bottom
        int x_base = (seed_origin == edges::east) * (N - 1);
        int y_base = (seed_origin == edges::south) * (N - 1);

        int x_offs = (seed_origin + 1) % 2;
        int y_offs = seed_origin % 2;

        for (int i = 0; i < PySequence_Size(seeds); ++i) {

            int seg_start;
            int seg_end;
            PyObject* segment = PySequence_GetItem(seeds, i);

#ifdef HEAVY_DEBUG
            assert(PyTuple_CheckExact(segment));
            assert(PySequence_Size(segment) == 2);
#endif
            if (!PyArg_ParseTuple(segment, "ii", &seg_start, &seg_end)) {
                Py_DECREF(segment);
                continue;
            }
            Py_DECREF(segment);
#ifdef HEAVY_DEBUG
            assert(segment->ob_refcnt == 1);
#endif

            bool contig = false;

            // Check all pixels in the segment, adding the first
            // of every contiguous section to the queue
            for (int n = seg_start, x = x_base + x_offs * n,
                     y = y_base + y_offs * n;
                 n <= seg_end; ++n, x += x_offs, y += y_offs) {

                input_seeds[n] = true; // mark incoming segment to skip reseed

                if (!dst(x, y) && pixel_fill_alpha(src(x, y)) > 0) {
                    if (!contig) {
                        queue.push(coord(x, y));
                        contig = true;
                    }
                } else {
                    contig = false;
                }
            }
        }
    } // Seed queue populated

    // 0-initialized arrays used to mark points reached on
    // the tile boundaries to potentially create new seed segments.
    // Ordered left-to-right / top-to-bottom, for n/s, e/w respectively
    bool edge_n[N] =
        {
            0,
        },
         edge_e[N] =
             {
                 0,
             },
         edge_s[N] =
             {
                 0,
             },
         edge_w[N] = {
             0,
         };
    bool* edge_marks[] = {edge_n, edge_e, edge_s, edge_w};

    while (!queue.empty()) {

        int x0 = queue.front().x;
        int y = queue.front().y;

        queue.pop();

        // skip if we're outside the bbox range
        if (y < min_y || y > max_y) continue;

        for (int i = 0; i < 2; ++i) {
            bool look_above = true;
            bool look_below = true;

            const int x_start =
                x0 + i; // include starting coordinate when moving left
            const int x_delta = i * 2 - 1; // first scan left, then right

            PixelRef<rgba> src_px = src.get_pixel(x_start, y);
            PixelRef<chan_t> dst_px = dst.get_pixel(x_start, y);

            for (int x = x_start; x >= min_x && x <= max_x;
                 x += x_delta, src_px.move_x(x_delta), dst_px.move_x(x_delta)) {
                if (dst_px.read()) {
                    break;
                } // Pixel is already filled

                chan_t alpha = pixel_fill_alpha(src_px.read());

                if (alpha <= 0) // Colors are too different
                {
                    break;
                }

                dst_px.write(alpha); // Fill the pixel

                if (y > 0) {
                    look_above = check_enqueue( //check/enqueue above
                        x, y-1, look_above, src_px.above(), dst_px.above());
                } else {
                    edge_n[x] = true; // On northern edge
                }
                if (y < (N - 1)) {
                    look_below = check_enqueue( // check/enqueue below
                        x, y+1, look_below, src_px.below(), dst_px.below());
                } else {
                    edge_s[x] = true; // On southern edge
                }

                if (x == 0) {
                    edge_w[y] = true; // On western edge
                } else if (x == (N - 1)) {
                    edge_e[y] = true; // On eastern edge
                }
            }
        }
    }

    if (seed_origin !=
        edges::none) { // Remove incoming seeds from outgoing seeds
        bool* edge = edge_marks[seed_origin];
        for (int n = 0; n < N; ++n) {
            edge[n] = edge[n] && !input_seeds[n];
        }
    }

    return Py_BuildValue(
        "[NNNN]", to_seeds(edge_n), to_seeds(edge_e), to_seeds(edge_s),
        to_seeds(edge_w));
}

void
Filler::flood(PyObject* src_arr, PyObject* dst_arr)
{
    PixelRef<rgba> src_px = PixelBuffer<rgba>(src_arr).get_pixel(0, 0);
    PixelRef<chan_t> dst_px = PixelBuffer<chan_t>(dst_arr).get_pixel(0, 0);
    for (int i = 0; i < N * N; ++i, src_px.move_x(1), dst_px.move_x(1)) {
        dst_px.write(pixel_fill_alpha(src_px.read()));
    }
}

// Gap closing

static inline PyObject*
simple_seeds(chan_t seeds[], edge e)
{
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
    Gap closing queue triple (x, y, distance)
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

static void
queue_gc_seeds(
    std::queue<gc_coord>& queue, gc_coord& c, chan_t curr_dist, chan_t north[],
    chan_t east[], chan_t south[], chan_t west[])
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
        // direction
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

PyObject*
rgba_tile_from_alpha_tile(
    PyObject* src, double fill_r, double fill_g, double fill_b, int min_x,
    int min_y, int max_x, int max_y)
{
    npy_intp dims[] = {N, N, 4};
    PyObject* dst_arr = PyArray_ZEROS(3, dims, NPY_USHORT, 0);
    PixelBuffer<rgba> dst_buf(dst_arr);
    PixelBuffer<chan_t> src_buf(src);
    for (int y = min_y; y <= max_y; ++y) {
        int x = min_x;
        PixelRef<chan_t> src_px = src_buf.get_pixel(x, y);
        PixelRef<rgba> dst_px = dst_buf.get_pixel(x, y);
        for (; x <= max_x; ++x, src_px.move_x(1), dst_px.move_x(1)) {
            dst_px.write(rgba(fill_r, fill_g, fill_b, src_px.read()));
        }
    }
    return dst_arr;
}
