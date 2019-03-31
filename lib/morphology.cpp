/* This file is part of MyPaint.
 * Copyright (C) 2018 by the MyPaint Development Team.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */
#include "morphology.hpp"
#include <cmath>

#include <numpy/ndarraytypes.h>

/*
  Helper function to copy a rectangular slice of the input
  buffer to the full input array.
*/
template <typename T>
static void fill_input_section(
    const int x, const int w,
    const int y, const int h,
    PixelBuffer<T> input_buf, T **input,
    const int px_x, const int px_y)
{
    PixelRef<T> in_px = input_buf.get_pixel(px_x, px_y);
    for (int y_i = y; y_i < y + h; ++y_i) {
        for (int x_i = x; x_i < x + w; ++x_i) {
            input[y_i][x_i] = in_px.read();
            in_px.move_x(1);
        }
        in_px.move_x(0-w);
        in_px.move_y(1);
    }
}

template <typename T> // The type of value morphed
static void copy_nine_grid_section(
    int radius, T **input, bool from_above,
    PyObject *mid,
    PyObject *n, PyObject *e,
    PyObject *s, PyObject *w,
    PyObject *ne, PyObject *se,
    PyObject *sw, PyObject *nw)
{
    const int r = radius;

    typedef PixelBuffer<T> PBT;

    if(from_above) {
        // Reuse radius*2 rows from previous morph
        // and no need to handle the topmost tiles
        for(int i = 0; i < r*2; ++i) {
            T *tmp = input[i];
            input[i] = input[N+i];
            input[N+i] = tmp;
        } // west, mid, east - partial
        fill_input_section<T>(0, r, 2*r, N-r, PBT(w), input, N-r, r);
        fill_input_section<T>(r, N, 2*r, N-r, PBT(mid), input, 0, r);
        fill_input_section<T>(N+r, r, 2*r, N-r, PBT(e), input, 0, r);
    }
    else { // nw, north, ne
        fill_input_section<T>(0, r, 0, r, PBT(nw), input, N-r, N-r);
        fill_input_section<T>(r, N, 0, r, PBT(n), input, 0, N-r);
        fill_input_section<T>(N+r, r, 0, r, PBT(ne), input, 0, N-r);

        // west, mid, east
        fill_input_section<T>(0, r, r, N, PBT(w), input, N-r, 0);
        fill_input_section<T>(r, N, r, N, PBT(mid), input, 0, 0);
        fill_input_section<T>(N+r, r, r, N, PBT(e), input, 0, 0);
    }
    // sw, south, se
    fill_input_section<T>(0, r, N+r, r, PBT(sw), input, N-r, 0);
    fill_input_section<T>(r, N, N+r, r, PBT(s), input, 0, 0);
    fill_input_section<T>(N+r, r, N+r, r, PBT(se), input, 0, 0);
}


MorphBucket::MorphBucket(int radius) :
    radius(radius), height(radius*2 + 1), se_chords (height)
{
    // Create structuring element

    int fst_length = 1 + 2 * floor(sqrt(powf((radius+0.5),2) - powf(radius,2)));

    for(int pad = 1; pad < fst_length; pad*=2)
    {
        se_lengths.push_back(pad);
    }
    // Go through the first half of the circle and populate the indices,
    // adding new unique chords as necessary
    for(int y = -radius; y <= 0; ++y) {
        int x_offs = floor(sqrt(powf((radius+0.5),2) - powf(y,2)));
        int length = 1+ x_offs * 2;
        if(se_lengths.back() != length)
            se_lengths.push_back(length);

        se_chords[y+radius] = chord(0-x_offs, se_lengths.size() - 1);
    }

    // Copy the mirrored indices from the first half to the second
    for(int mirr_y = 1; mirr_y <= radius; mirr_y++) {
        se_chords[mirr_y + radius] = se_chords[(0 - mirr_y) + radius];
    }

    const int width = N + 2*radius;

    // Allocate input space
    input = new chan_t*[width];
    for(int i = 0; i < width; ++i) {
        input[i] = new chan_t[width];
    }
    // Allocate lookup table
    const int num_types = se_lengths.size();
    table = new chan_t**[height];
    for(int h = 0; h < height; ++h) {
        table[h] = new chan_t*[width];
        for(int w = 0; w < width; ++w) {
            table[h][w] = new chan_t[num_types];
        }
    }
}
MorphBucket::~MorphBucket()
{
    const int width = N + 2*radius;

    // Free input
    for(int i = 0; i < width; ++i) {
        delete[] input[i];
    }
    delete[] input;

    // Free lookup table
    for(int h = 0; h < height; ++h) {
        for(int w = 0; w < width; ++w) {
            delete[] table[h][w];
        }
        delete table[h];
    }
    delete[] table;
}


/*
  Rotate the lookup table down one step
*/
void MorphBucket::rotate_lut()
{
    chan_t **first = table[0];
    for(int y = 0; y < height-1; ++y) {
        table[y] = table[y+1];
    }
    table[height-1] = first;
}

template <op cmp>
void MorphBucket::populate_row(int y_row, int y_px)
{
    const int r = radius;

    for(int x = 0; x < N + 2*r; ++x) {
        table[y_row][x][0] = input[y_px][x];
    }
    int prev_len = 1;
    for(int len_i = 1; len_i < se_lengths.size(); len_i++) {
        const int len = se_lengths[len_i];
        const int len_diff = len - prev_len;
        prev_len = len;
        for(int x = 0; x <= N + 2*r - len; ++x) {
            chan_t ext_v = cmp(table[y_row][x][len_i - 1],
                               table[y_row][x+len_diff][len_i - 1]);
            table[y_row][x][len_i] = ext_v; // Consider changing access order
        }
    }
}

/*
  Search the diameter of a circle (cx, cy, w) horizontally
  and vertically for any pixel equalling the limit value.
*/
static bool check_lim(chan_t lim, PixelBuffer<chan_t> &buf, int cx, int cy, int w)
{
    for(int y = 0; y <= 1; ++y) {
        for(int x = - w; x <= w; ++x) {
            if(buf(cx+x,cy+y) == lim ||
               buf(cx+y,cy+x) == lim) {
                return true;
            }
        }
    }
    return false;
}

/*
  Search a disjunction (or conjunction of disjunctions) of pixels
  for the limit value, if the radius is large enough to cover the
  entire tile if a limit valued pixel (or pixels) is found.
*/
template <chan_t lim>
bool MorphBucket::can_skip(PixelBuffer<chan_t> buf)
{
    const int r = radius;
    const int max_search_radius = 15;
    const int r_limit = (N * sqrt(2)) / 2;

    // Structuring element covers the entire tile
    if(r > r_limit) {
        int range = MIN((r - r_limit), max_search_radius);
        const int half = N/2 - 1;
        if (check_lim(lim, buf, half, half, range)) {
            return true;
        }
    }
    // Four structuring elements can cover the tile
    if(r > (r_limit / 2)) {
        int range = MIN(r - (r_limit / 2), max_search_radius);
        const int qrtr = N/4;
        const int r_px = - 1;
        if (check_lim(lim, buf, r_px + qrtr, r_px + qrtr, range) && // nw
            check_lim(lim, buf, r_px + 3*qrtr, r_px + qrtr, range) && //ne
            check_lim(lim, buf, r_px + 3*qrtr, r_px + 3*qrtr, range) && //se
            check_lim(lim, buf, r_px + qrtr, r_px + 3*qrtr, range)) //sw
        {
            return true;
        }
    }

    return false;
}

template <chan_t init, chan_t lim, op cmp>
void MorphBucket::morph(bool can_update, PixelBuffer<chan_t> &dst)
{
    const int r = radius;

    if(can_update)
    {
        populate_row<cmp>(0, 2*radius);
        rotate_lut();
    }
    else
    {
        for(int dy = 0; dy < height; ++dy) {
            populate_row<cmp>(dy, dy);
        }

    }
    PixelRef<chan_t> dst_px = dst.get_pixel(0,0);
    for(int y = 0; y < N; ++y) {
        for(int x = 0; x < N; ++x) {
            chan_t ext = init;
            for(int c = 0; c < height; ++c) {
                chord &ch = se_chords[c];
                ext = cmp(ext, table[c][x + ch.x_offset + r][ch.length_index]);
                if(ext == lim)
                    break;
            }
            dst_px.write(ext);
            dst_px.move_x(1);
        }
        if(y < N-1)
        {
            populate_row<cmp>(0, y + 2*radius + 1);
            rotate_lut();
        }
    }
}

void MorphBucket::initiate(
    bool can_update,
    PyObject *mid,
    PyObject *n, PyObject *e,
    PyObject *s, PyObject *w,
    PyObject *ne, PyObject *se,
    PyObject *sw, PyObject *nw)
{
    copy_nine_grid_section(
        radius, input, can_update,
        mid, n, e, s, w,
        ne, se, sw, nw);
}

template <chan_t init, chan_t lim, op cmp>
static PyObject* generic_morph(
    MorphBucket &mb,
    bool can_update,
    PyObject *mid,
    PyObject *n, PyObject *e,
    PyObject *s, PyObject *w,
    PyObject *ne, PyObject *se,
    PyObject *sw, PyObject *nw)
{

    mb.initiate(can_update, mid,
                n, e, s, w,
                ne, se, sw, nw);

    if (mb.can_skip<lim>(PixelBuffer<chan_t>(mid))) {
        return Py_BuildValue("(b())", false);
    }
    else {
        npy_intp dims[] = {N, N};
        PyObject* dst_tile = PyArray_EMPTY(2, dims, NPY_USHORT, 0);

        PixelBuffer<chan_t> dst_buf = PixelBuffer<chan_t> (dst_tile);
        mb.morph<init, lim, cmp>(
            can_update, dst_buf);
        PyObject* result = Py_BuildValue("(bN)", true, dst_tile);
#ifdef HEAVY_DEBUG
        assert(dst_tile->ob_refcnt == 1);
#endif
        return result;
    }
}

inline chan_t max(chan_t a, chan_t b)
{
    return a > b ? a : b;
}


inline chan_t min(chan_t a, chan_t b)
{
    return a < b ? a : b;
}

PyObject* dilate(
    MorphBucket &mb,
    bool can_update,
    PyObject *mid,
    PyObject *n, PyObject *e,
    PyObject *s, PyObject *w,
    PyObject *ne, PyObject *se,
    PyObject *sw, PyObject *nw)
{
    return generic_morph<0, fix15_one, max>(
        mb, can_update, mid,
        n, e, s, w,
        ne, se, sw, nw);
}

PyObject* erode(
    MorphBucket &mb,
    bool can_update,
    PyObject *mid,
    PyObject *n, PyObject *e,
    PyObject *s, PyObject *w,
    PyObject *ne, PyObject *se,
    PyObject *sw, PyObject *nw)
{
    return generic_morph<fix15_one, 0, min>(
        mb, can_update, mid,
        n, e, s, w,
        ne, se, sw, nw);
}


// Box blur parameters & memory allocation

BlurBucket::BlurBucket(int radius) : radius (radius), height (radius * 2 + 1)
{
    // Suppress uninitialization warning, the output array is always
    // fully written before use
    output[0][0] = 0;
    const int d = N + radius * 2;
    // Output from 3x3-grid,
    // input to horizontal blur (Y x X) = (d x d)
    input_full = new chan_t*[d];
    for(int i = 0; i < d; ++i) {
        input_full[i] = new chan_t[d];
    }
    // Output for horizontal blur,
    // input to vertical blur (Y x X) = (d x N)
    input_vert = new chan_t*[d];
    for(int i = 0; i < d; ++i) {
        input_vert[i] = new chan_t[N];
    }
}

BlurBucket::~BlurBucket()
{
    const int d = N + radius * 2;
    for(int i = 0; i < d; ++i) {
        delete[] input_full[i];
        delete[] input_vert[i];
    }
    delete[] input_full;
    delete[] input_vert;
}

/*
  Blur (N + 2 * radius) rows horizontally (running average)
*/
static void blur_hor(int radius, chan_t **input, chan_t **output)
{
    const int w = 2 * radius + 1;
    const fix15_t dvs = (fix15_t) (w * fix15_one);

    // Traverse N + 2 * radius lines of the input and write to output
    for(int y = 0; y < N + 2 * radius; ++y) {

        // Calculate average of first radius*2+1 values
        fix15_t avg = 0;
        for(int i = 0; i < w; i++) { avg += fix15_div(input[y][i], dvs); }
        output[y][0] = fix15_short_clamp(avg);
        int bot = 0;
        int top = w;
        for(int x = 1; x < N; ++x) { // step through the row
            const chan_t bot_v = fix15_div(input[y][bot], dvs);
            const chan_t top_v = fix15_div(input[y][top], dvs);
            avg = avg - bot_v + top_v;
            output[y][x] = fix15_short_clamp(avg);
            bot++;
            top++;
        }
    }
}

/*
  Blur N rows horizontally (running average)
*/
static void blur_ver(int radius, chan_t **input, chan_t output[N][N])
{
    const int h = radius * 2 + 1;
    const fix15_t dvs = (fix15_t) (h * fix15_one);

    // Traverse horizontally here as well, for memory locality
    for(int x = 0; x < N; ++x) {
        fix15_t avg = 0;
        for(int y = 0; y < h; ++y) {
            avg += fix15_div(input[y][x], dvs);
        }
        output[0][x] = avg;
    }
    int bot = 0;
    int top = h;
    for(int y = 1; y < N; ++y) {
        for(int x = 0; x < N; ++x) {
            const chan_t bot_v = fix15_div(input[bot][x], dvs);
            const chan_t top_v = fix15_div(input[top][x], dvs);
            output[y][x] = output[y-1][x] - bot_v + top_v;
        }
        bot++;
        top++;
    }
}

/*
  Perform a box blur using 3x3 input tiles, writing the blurred
  pixels to the destination tile.

  The can_update parameter is used to skip redundant pixel transfers
  between consecutive tiles (when going top-to-bottom).
*/
void blur(BlurBucket &bb, bool can_update,
          PyObject *mid, PyObject* dst_tile,
          PyObject *n, PyObject *e,
          PyObject *s, PyObject *w,
          PyObject *ne, PyObject *se,
          PyObject *sw, PyObject *nw)
{
    copy_nine_grid_section(
        bb.radius, bb.input_full, can_update,
        mid, n, e, s, w,
        ne, se, sw, nw);

    blur_hor(bb.radius, bb.input_full, bb.input_vert);
    blur_ver(bb.radius, bb.input_vert, bb.output);

    PixelRef<chan_t> dst_px =
        PixelBuffer<chan_t>(dst_tile).get_pixel(0,0);

    for(int y = 0; y < N; ++y) {
        for(int x = 0; x < N; ++x) {
            dst_px.write(bb.output[y][x]);
            dst_px.move_x(1);
        }
    }
}


DistanceBucket::DistanceBucket(int distance) : distance (distance)
{
    int r = N + distance * 2 + 2;
    input = new chan_t*[r];
    for (int i = 0; i < r; ++i)
        input[i] = new chan_t[r];
}

DistanceBucket::~DistanceBucket()
{
    int r = N + distance * 2 + 2;
    for (int i = 0; i < r; ++i)
        delete[] input[i];
    delete[] input;
}

typedef coord (*rot_op)(int x, int y, int xoff, int yoff);

static inline void
upd_dist(coord lc, PixelBuffer<chan_t> &dists, int new_dst) {
    if (lc.x < 0 || lc.x > (N-1) || lc.y < 0 || lc.y > (N-1))
        return;
    int curr_dist = dists(lc.x, lc.y);
    if (curr_dist > new_dst) {
        dists(lc.x, lc.y) = new_dst;
    }
}

// Search an octant with a radius of _dist_ pixels, marking any gaps
// that are found. The octant searched is determined by the rotation
// function provided.
void dist_search(int x, int y, int dist,
                chan_t **alphas, PixelBuffer<chan_t> &dists,
                rot_op op)
{

    //int d_lim = 1 + (dist * dist);
    int offs = dist + 1;
    int rx = x - offs;
    int ry = y - offs;

    coord t1 = op(x, y, 0, -1);
    coord t2 = op(x, y, 1, -1);

    if (alphas[t1.y][t1.x] == 0 || alphas[t2.y][t2.x] == 0)
        return;

    for(int yoffs = 2; yoffs < dist + 2; ++yoffs) {
        int y_dst_sqr = (yoffs-1)*(yoffs-1);

        for(int xoffs = 0; xoffs <= yoffs; ++xoffs) {
            int offs_dst = y_dst_sqr + (xoffs)*(xoffs);
            //int real_offs = (int) round(sqrtf((float) offs_dst));
            if (offs_dst >= 1 + dist * dist)
                break;
            coord c = op(x, y, xoffs, -yoffs);
            if(alphas[c.y][c.x] == 0) { // Gap found

                // Double-width distance assignment
                float dx = (float) xoffs / (yoffs - 1);
                float tx = 0;
                int cx = 0;
                for(int cy = 1; cy < yoffs; ++cy) {
                    upd_dist(op(rx, ry, cx, 0-cy), dists, offs_dst);
                    tx += dx;
                    if (floor(tx) > cx){
                        cx++;
                        upd_dist(op(rx, ry, cx, 0-cy), dists, offs_dst);
                    }
                    upd_dist(op(rx, ry, cx+1, 0-cy), dists, offs_dst);
                }
            }
        }
    }
}

// Coordinate reflection/rotation
coord top_right(int x, int y, int xoffs, int yoffs) { return coord(x + xoffs, y + yoffs); }
coord top_centr(int x, int y, int xoffs, int yoffs) { return coord(x - yoffs, y - xoffs); }
coord bot_centr(int x, int y, int xoffs, int yoffs) { return coord(x - yoffs, y + xoffs); }
coord bot_right(int x, int y, int xoffs, int yoffs) { return coord(x + xoffs, y - yoffs); }


/* Search for gaps in the 9-grid of flooded alpha tiles,
   a gap being defined as a
 */
void find_gaps(
    DistanceBucket &rb,
    PyObject *radiuses_arr,
    PyObject *mid,
    PyObject *n,
    PyObject *e,
    PyObject *s,
    PyObject *w,
    PyObject *ne,
    PyObject *se,
    PyObject *sw,
    PyObject *nw)
{
    int r = rb.distance + 1;

    copy_nine_grid_section(
        r, rb.input, false,
        mid, n, e, s, w,
        ne, se, sw, nw);

    PixelBuffer<chan_t> radiuses (radiuses_arr);
    // search for gaps in an approximate semi-circle
    for (int y = 0; y < 2*r + N-1; ++y) { // we check at most distance+1 pixels above any point
        for (int x = 0; x < r + N-1; ++x) {
            if(rb.input[y][x] == 0) { // Search for gaps in relation to this pixel
                if (y >= r) {
                    dist_search(x, y, rb.distance, rb.input, radiuses, top_right);
                    dist_search(x, y, rb.distance, rb.input, radiuses, top_centr);
                }
                if (y < N + r) {
                    dist_search(x, y, rb.distance, rb.input, radiuses, bot_centr);
                    dist_search(x, y, rb.distance, rb.input, radiuses, bot_right);
                }
            }
        }
    }
}

static inline bool any_unfillable(
    int x, const int w,
    int y, const int h,
    PixelBuffer<chan_t> tile)
{
    int xlim = x + w;
    int ylim = y + h;
    PixelRef<chan_t> px = tile.get_pixel(x, y);
    for (; y < ylim; ++y) {
        for (; x < xlim; ++x) {
            if (px.read() == 0) {
                return true;
            }
            px.move_x(1);
        }
        px.move_x(0-w);
        px.move_y(1);
    }
    return false;
}

// Checks corners of tiles assumed to be adjacent to an empty tile
// in the middle of them, return true if any unfillable pixels are
// found that may cause distance data to be required for the middle tile
bool no_corner_gaps(
    int d, // distance
    PyObject *n,
    PyObject *e,
    PyObject *s,
    PyObject *w)
{
    PixelBuffer<chan_t> north = PixelBuffer<chan_t>(n);
    PixelBuffer<chan_t> east = PixelBuffer<chan_t>(e);
    PixelBuffer<chan_t> south = PixelBuffer<chan_t>(s);
    PixelBuffer<chan_t> west = PixelBuffer<chan_t>(w);

    //NE corner of W tile, check SW of N if any found
    if (any_unfillable(N-d, d, 0, d, west))
        if (any_unfillable(0, d, N-d, d, north))
            return false;

    //SE corner of W tile, check NW of S if any found
    if (any_unfillable(N-d, d, N-d, d, west))
        if (any_unfillable(0, d, 0, d, south))
            return false;

    //SE corner of N tile, check NW of E if any found
    if (any_unfillable(N-d, d, N-d, d, north))
        if (any_unfillable(0, d, 0, d, east))
            return false;

    //NE corner of S tile, check SW of E if any found
    if (any_unfillable(N-d, d, 0, d, south))
        if (any_unfillable(0, d, N-d, d, east))
            return false;

    // No crorner-crossing gaps possible
    return true;
}
