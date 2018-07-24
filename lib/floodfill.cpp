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

#include <numpy/ndarraytypes.h>

#include <mypaint-tiled-surface.h>

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
    PyObject *seg_list = PyList_New(0);

    bool contiguous = false;
    int seg_start = 0;
    int seg_end = 0;

    for(int c = 0; c < N; ++c)
    {
        if(edge[c]) {
            if(!contiguous) { // a new segment begins
                seg_start = c;
                seg_end = c;
                contiguous = true;
            }
            else { // segment continues
                seg_end++;
            }
        }
        else if (contiguous) { // segment ends
            PyObject *segment = Py_BuildValue("ii", seg_start, seg_end);
            PyList_Append(seg_list, segment);
            Py_DECREF(segment);
            contiguous = false;
        }
    }
    if (contiguous) {
        PyObject *segment = Py_BuildValue("ii", seg_start, seg_end);
        PyList_Append(seg_list, segment);
        Py_DECREF(segment);
    }

    return seg_list;
}

static inline chan_t
clamped_div(chan_t a, chan_t b)
{
    return fix15_short_clamp(
        fix15_div(fix15_short_clamp(a), b));
}

static inline rgba
straightened(int targ_r, int targ_g,
             int targ_b, int targ_a)
{
    if (targ_a <= 0)
        return rgba((chan_t) 0, 0, 0, 0);
    else
        return rgba(clamped_div(targ_r, targ_a),
                    clamped_div(targ_g, targ_a),
                    clamped_div(targ_b, targ_a),
                    targ_a);
}

Filler::Filler(int targ_r, int targ_g,
                         int targ_b, int targ_a, double tol) :
    targ (straightened(targ_r, targ_g, targ_b, targ_a)),
    targ_premult (rgba((chan_t) targ_r, targ_g, targ_b, targ_a)),
    tolerance ((fix15_t)(MIN(1.0, MAX(0.0, tol)) * fix15_one))
{ }

/*
  Evaluates the source pixel based on the target color and
  tolerance value, returning the alpha of the fill as it
  should be set for the corresponding destination pixel.

  A non-zero result indicates that the destination pixel
  should be filled.
*/
chan_t Filler::pixel_fill_alpha(const rgba &px)
{
    fix15_t dist;

    if((targ.alpha | px.alpha) == 0)
        return fix15_one;
    else if (tolerance == 0)
    {
        return fix15_one * (targ_premult == px);
    }

    if (targ.alpha == 0)
        dist = px.alpha;
    else
        dist = targ.max_diff(straightened(px.red, px.green, px.blue, px.alpha));

    // Compare with adjustable tolerance of mismatches.
    static const fix15_t onepointfive = fix15_one + fix15_halve(fix15_one);
    dist = fix15_div(dist, tolerance);
    if (dist > onepointfive) {  // aa < 0, but avoid underflow
        return 0;
    }
    else {
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
bool Filler::check_enqueue(
    const int x, const int y, bool check,
    const rgba &src_pixel,
    const chan_t &dst_pixel)
{
    if (dst_pixel != 0)
        return true;
    bool match = pixel_fill_alpha(src_pixel) > 0;
    if(match && check) {
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
PyObject * Filler::tile_uniformity(bool empty_tile, PyObject *src_arr)
{
    if(empty_tile) {
        chan_t alpha = pixel_fill_alpha(rgba((chan_t) 0, 0, 0, 0));
        return Py_BuildValue("i", alpha);
    }

    PixelBuffer<rgba> src = PixelBuffer<rgba>((PyArrayObject*) src_arr);

    if(src.is_uniform()) {
        chan_t alpha = pixel_fill_alpha(src(0,0));
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
PyObject* Filler::fill(
    PyObject *src_o,
    PyObject *dst_o,
    PyObject *seeds,
    edge seed_origin,
    int min_x, int min_y, int max_x, int max_y)
{
    if (min_x > max_x || min_y > max_y)
        return Py_BuildValue("[()()()()]");
    if (min_x < 0) min_x = 0;
    if (min_y < 0) min_y = 0;
    if (max_x > (N-1)) max_x = (N-1);
    if (max_y > (N-1)) max_y = (N-1);

    PixelBuffer<rgba> src (src_o);
    PixelBuffer<chan_t> dst (dst_o);

    // Store input seed positions to filter them out
    // prior to constructing the output seed segment lists
    bool input_seeds[N] = {0,};

    if(PyTuple_CheckExact(seeds)) { // the very first seed
        coord seed_pt;
        PyArg_ParseTuple(seeds, "ii", &(seed_pt.x), &(seed_pt.y));
        queue.push(seed_pt);
    }
    else {
        // The tile edge and direction determined by where the seed segments
        // originates, left-to-right or top-to-bottom
        int x_base = (seed_origin == edges::east) * (N-1);
        int y_base = (seed_origin == edges::south) * (N-1);

        int x_offs = (seed_origin + 1) % 2;
        int y_offs = seed_origin % 2;

        for (int i=0; i<PySequence_Size(seeds); ++i) {

            int seg_start;
            int seg_end;
            PyObject *segment = PySequence_GetItem(seeds, i);
            if (! PyArg_ParseTuple(segment,"ii", &seg_start, &seg_end)) {
                Py_DECREF(segment);
                continue;
            }
            Py_DECREF(segment);

            bool contig = false;

            // Check all pixels in the segment, adding the first
            // of every contiguous section to the queue
            for(int n = seg_start,
                    x = x_base + x_offs * n,
                    y = y_base + y_offs * n;
                n <= seg_end;
                ++n, x += x_offs, y += y_offs) {

                input_seeds[n] = true; // mark incoming segment to skip reseed

                if(!dst(x, y) && pixel_fill_alpha(src(x, y)) > 0) {
                    if (!contig) {
                        queue.push(coord(x, y));
                        contig = true;
                    }
                }
                else {
                    contig = false;
                }
            }
        }
    } // Seed queue populated


    // 0-initialized arrays used to mark points reached on
    // the tile boundaries to potentially create new seed segments.
    // Ordered left-to-right / top-to-bottom, for n/s, e/w respectively
    bool edge_n[N] = {0,}, edge_e[N] = {0,},
         edge_s[N] = {0,}, edge_w[N] = {0,};
    bool *edge_marks[] = {edge_n, edge_e, edge_s, edge_w};

    while (! queue.empty()) {

        int x0 = queue.front().x;
        int y = queue.front().y;

        queue.pop();

	// skip if we're outside the bbox range
	if (y < min_y || y > max_y)
            continue;

        for (int i=0; i<2; ++i)
        {
            bool look_above = true;
            bool look_below = true;

            const int x_start = x0 + i; // include starting coordinate when moving left
            const int x_delta = i*2 - 1; // first scan left, then right

            PixelRef<rgba> src_px = src.get_pixel(x_start, y);
            PixelRef<chan_t> dst_px = dst.get_pixel(x_start, y);

            for ( int x = x_start;
                  x >= min_x && x <= max_x
                      ; x += x_delta,
                      src_px.move_x(x_delta),
                      dst_px.move_x(x_delta)
	        )
            {
                if(dst_px.read()) { break; } // Pixel is already filled

                chan_t alpha = pixel_fill_alpha(src_px.read());

                if(alpha <= 0) // Colors are too different
                {
                    dst_px.write(0x0001); // Mark with nearly transparent alpha
                    break;
                }

                dst_px.write(alpha); // Fill the pixel

                if(y > 0) {
                    look_above = check_enqueue( //check/enqueue above
                        x, y-1, look_above, src_px.above(), dst_px.above());
                }
                else {
                    edge_n[x] = true; // On northern edge
                }
                if(y < (N-1)) {
                    look_below = check_enqueue( // check/enqueue below
                        x, y+1, look_below, src_px.below(), dst_px.below());
                }
                else {
                    edge_s[x] = true; // On southern edge
                }

                if(x == 0) {
                    edge_w[y] = true; // On western edge
                }
                else if(x == (N-1)) {
                    edge_e[y] = true; // On eastern edge
                }
            }
        }
    }

    if(seed_origin != edges::none) { // Remove incoming seeds from outgoing seeds
      bool *edge = edge_marks[seed_origin];
        for(int n = 0; n < N; ++n) {
            edge[n] = edge[n] && !input_seeds[n];
        }
    }

    return Py_BuildValue("[NNNN]",
                         to_seeds(edge_n), to_seeds(edge_e),
                         to_seeds(edge_s), to_seeds(edge_w));
}

void Filler::flood(PyObject *src_arr, PyObject *dst_arr)
{
    PixelRef<rgba> src_px = PixelBuffer<rgba>(src_arr).get_pixel(0,0);
    PixelRef<chan_t> dst_px = PixelBuffer<chan_t>(dst_arr).get_pixel(0,0);
    for(int i = 0; i < N*N; ++i, src_px.move_x(1), dst_px.move_x(1)) {
        dst_px.write(pixel_fill_alpha(src_px.read()));
    }
}

PyObject* full_rgba_tile(double fill_r, double fill_g, double fill_b)
{
    npy_intp dims[] = {N, N, 4};
    PyObject* dst_arr = PyArray_EMPTY(3, dims, NPY_USHORT, 0);
    PixelBuffer<rgba> dst ((PyArrayObject*) dst_arr);
    PixelRef<rgba> px = dst.get_pixel(0, 0);
    rgba color = rgba(fill_r, fill_g, fill_b, fix15_one);
    for(int n = 0; n < TILE_SQUARED; ++n, px.move_x(1)) {
        px.write(color);
    }
    return dst_arr;
}


static void comp_px(double fill_r, double fill_g, double fill_b,
             PixelRef<chan_t> &src_px, PixelRef<rgba> &dst_px)
{
    if (src_px.read() != 0x0001)
    {
        const rgba &d = dst_px.read();
        const fix15_t as = src_px.read();
        const fix15_t j = fix15_one - as;
        const fix15_t k = fix15_mul(d.alpha, j);
        const fix15_short_t r = fix15_short_clamp(fill_r * fix15_one);
        const fix15_short_t g = fix15_short_clamp(fill_g * fix15_one);
        const fix15_short_t b = fix15_short_clamp(fill_b * fix15_one);
        const rgba result = rgba(
            fix15_short_clamp(fix15_sumprods(as, r, j, d.red)),
            fix15_short_clamp(fix15_sumprods(as, g, j, d.green)),
            fix15_short_clamp(fix15_sumprods(as, b, j, d.blue)),
            fix15_short_clamp(as + k)
            );
        dst_px.write(result);
    }
}


void fill_composite(
    double fill_r, double fill_g, double fill_b,
    PyObject* src, PyObject *dst,
    int min_x, int min_y, int max_x, int max_y)
{
    PixelBuffer<chan_t> src_buf (src);
    PixelBuffer<rgba> dst_buf (dst);

    if(min_x == 0 && min_y == 0 && max_x == N-1 && max_y == N-1) {

        PixelRef<chan_t> src_px = src_buf.get_pixel(0,0);
        PixelRef<rgba> dst_px = dst_buf.get_pixel(0,0);

        for(int i = 0; i < TILE_SQUARED; i++, src_px.move_x(1), dst_px.move_x(1))
        {
            comp_px(fill_r, fill_g, fill_b, src_px, dst_px);
        }
    }
    else {
        for(int y = min_y; y <= max_y; ++y)
        {
            int x = min_x;
            PixelRef<chan_t> src_px = src_buf.get_pixel(x, y);
            PixelRef<rgba> dst_px = dst_buf.get_pixel(x, y);
            for(; x <= max_x; ++x, src_px.move_x(1), dst_px.move_x(1))
            {
                comp_px(fill_r, fill_g, fill_b, src_px, dst_px);
            }
        }
    }
}
