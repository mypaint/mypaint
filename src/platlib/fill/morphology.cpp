/* This file is part of MyPaint.
 * Copyright (C) 2018 by the MyPaint Development Team.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */

#include "morphology.hpp"
#include "fill_constants.hpp"

#include <cmath>
#include <tuple>

/*
  Get the width of a horizontal chord, located at a given vertical distance
  from the center of the circle for a circular structuring element with the
  given radius. Note that the radius is not a radius the normal sense since
  the width always has to be an uneven number - the width is 2r + 1.
*/
static inline int
chord_width(int radius, int y_offset)
{
    // Simple derivation from the pythagorean theorem, adjusted to
    // achieve vertical and horizontal symmetry and ensuring that
    // padding only has to be generated up to the first chord.
    int hw = floor(sqrt(powf((radius + 0.5), 2) - powf(y_offset, 2)));
    return 1 + 2 * hw;
}

Morpher::Morpher(int radius)
    : radius(radius), height(radius * 2 + 1), se_chords(height)
{
    // Create structuring element

    int fst_length = chord_width(radius, radius);

    for (int pad = 1; pad < fst_length; pad *= 2) {
        se_lengths.push_back(pad);
    }
    // Go through the first half of the circle and populate the indices,
    // adding new unique chords as necessary
    for (int y = -radius; y <= 0; ++y) {
        int length = chord_width(radius, y);
        if (se_lengths.back() != length) se_lengths.push_back(length);
        int x_offset = (length - 1) / -2;
        se_chords[y + radius] = chord(x_offset, se_lengths.size() - 1);
    }

    // Copy the mirrored indices from the first half to the second
    for (int mirr_y = 1; mirr_y <= radius; mirr_y++) {
        se_chords[mirr_y + radius] = se_chords[(0 - mirr_y) + radius];
    }

    const int width = N + 2 * radius;

    // Allocate input space
    input = new chan_t*[width];
    for (int i = 0; i < width; ++i) {
        input[i] = new chan_t[width];
    }
    // Allocate lookup table
    const int num_types = se_lengths.size();
    lookup_table = new chan_t**[height];
    for (int h = 0; h < height; ++h) {
        lookup_table[h] = new chan_t*[width];
        for (int w = 0; w < width; ++w) {
            lookup_table[h][w] = new chan_t[num_types];
        }
    }
}
Morpher::~Morpher()
{
    const int width = N + 2 * radius;

    // Free input
    for (int i = 0; i < width; ++i) {
        delete[] input[i];
    }
    delete[] input;

    // Free lookup table
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            delete[] lookup_table[h][w];
        }
        delete lookup_table[h];
    }
    delete[] lookup_table;
}

/*
  Rotate the lookup table down one step
*/
void
Morpher::rotate_lut()
{
    chan_t** first = lookup_table[0];
    for (int y = 0; y < height - 1; ++y) {
        lookup_table[y] = lookup_table[y + 1];
    }
    lookup_table[height - 1] = first;
}

template <op cmp>
void
Morpher::populate_row(int y_row, int y_px)
{
    const int r = radius;

    for (int x = 0; x < N + 2 * r; ++x) {
        lookup_table[y_row][x][0] = input[y_px][x];
    }
    int prev_len = 1;
    for (size_t len_i = 1; len_i < se_lengths.size(); len_i++) {
        const int len = se_lengths[len_i];
        const int len_diff = len - prev_len;
        prev_len = len;
        for (int x = 0; x <= N + 2 * r - len; ++x) {
            chan_t ext_v =
                cmp(lookup_table[y_row][x][len_i - 1],
                    lookup_table[y_row][x + len_diff][len_i - 1]);
            lookup_table[y_row][x][len_i] = ext_v;
        }
    }
}

/*
  Search the diameter of a circle (cx, cy, w) horizontally
  and vertically for any pixel equalling the limit value.
*/
static bool
lim_search(chan_t lim, PixelBuffer<chan_t>& buf, int cx, int cy, int w)
{
    for (int y = 0; y <= 1; ++y) {
        for (int x = -w; x <= w; ++x) {
            if (buf(cx + x, cy + y) == lim || buf(cx + y, cy + x) == lim) {
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
bool
Morpher::can_skip(PixelBuffer<chan_t> buf)
{
    const int r = radius;
    const int max_search_radius = 15;
#define SQRT2 1.4142135623730951
    const int r_limit = (N * SQRT2) / 2;
#undef SQRT2

    // Structuring element covers the entire tile
    if (r > r_limit) {
        int range = MIN((r - r_limit), max_search_radius);
        const int half = N / 2 - 1;
        if (lim_search(lim, buf, half, half, range)) {
            return true;
        }
    }
    // Four structuring elements can cover the tile
    if (r > (r_limit / 2)) {
        int range = MIN(r - (r_limit / 2), max_search_radius);
        const int q = N / 4;
        const int r_px = -1;
        if (lim_search(lim, buf, r_px + q, r_px + q, range) && // nw
            lim_search(lim, buf, r_px + 3 * q, r_px + q, range) && // ne
            lim_search(lim, buf, r_px + 3 * q, r_px + 3 * q, range) && // se
            lim_search(lim, buf, r_px + q, r_px + 3 * q, range)) // sw
        {
            return true;
        }
    }

    return false;
}

template <chan_t init, chan_t lim, op cmp>
void
Morpher::morph(bool can_update, PixelBuffer<chan_t>& dst)
{
    const int r = radius;

    if (can_update) {
        populate_row<cmp>(0, 2 * radius);
        rotate_lut();
    } else {
        for (int dy = 0; dy < height; ++dy) {
            populate_row<cmp>(dy, dy);
        }
    }
    PixelRef<chan_t> dst_px = dst.get_pixel(0, 0);
    for (int y = 0; y < N; ++y) {
        for (int x = 0; x < N; ++x) {
            chan_t ext = init;
            for (int c = 0; c < height; ++c) {
                chord& ch = se_chords[c];
                ext = cmp(ext, lookup_table[c][x + ch.x_offset + r][ch.length_index]);
                if (ext == lim) break;
            }
            dst_px.write(ext);
            dst_px.move_x(1);
        }
        if (y < N - 1) {
            populate_row<cmp>(0, y + 2 * radius + 1);
            rotate_lut();
        }
    }
}

void
Morpher::initiate(bool can_update, GridVector grid)
{
    init_from_nine_grid(radius, input, can_update, grid);
}

/*
  Perform a morphological operation with the templated arguments
  for value extremes and the comparison operation
  (in practice, this is either (0, 1, >) or (1, 0, <))

  Returns a pair, where the first item indicates whether the input
  array can be partially updated for the subsequent tile, and the
  second item is a pointer to the (potentially new) tile resulting
  from the operation.
 */
template <chan_t init, chan_t lim, op cmp>
static std::pair<bool, PyObject*>
generic_morph(
    Morpher& mb, bool update_input, bool update_lut, GridVector input)
{
    // Run a quick check, only run for large radiuses
    if (mb.can_skip<lim>(input[4])) {
        if (lim == 0)
            return std::make_pair(false, ConstTiles::ALPHA_TRANSPARENT());
        else
            return std::make_pair(false, ConstTiles::ALPHA_OPAQUE());
    }

    mb.initiate(update_input, input);

    // Check the entire input before running an actual
    // morph, potentially avoids tile allocation + op
    // No big performance diff. for morph itself, but can
    // speed up subsequent ops (blur + compositing)
    if (mb.input_fully_transparent())
        return std::make_pair(true, ConstTiles::ALPHA_TRANSPARENT());
    else if (mb.input_fully_opaque())
        return std::make_pair(true, ConstTiles::ALPHA_OPAQUE());

    PixelBuffer<chan_t> dst_buf = new_alpha_tile();

    mb.morph<init, lim, cmp>(update_lut, dst_buf);

    return std::make_pair(true, dst_buf.array_ob);
}

inline chan_t
max(chan_t a, chan_t b)
{
    return a > b ? a : b;
}

inline chan_t
min(chan_t a, chan_t b)
{
    return a < b ? a : b;
}

std::pair<bool, PyObject*>
dilate(Morpher& mb, bool update_input, bool update_lut, GridVector input)
{
    return generic_morph<0, fix15_one, max>(
        mb, update_input, update_lut, input);
}

std::pair<bool, PyObject*>
erode(Morpher& mb, bool update_input, bool update_lut, GridVector input)
{
    return generic_morph<fix15_one, 0, min>(
        mb, update_input, update_lut, input);
}

// Morph a single strand of tiles, storing
// the output tiles in a Python dictionary "morphed"
void
morph_strand(
    int offset, // Dilation/erosion radius (+/-)
    Strand& strand, AtomicDict tiles, Morpher& bucket, AtomicDict morphed,
    Controller& status_controller)
{
    auto op = offset > 0 ? dilate : erode;
    bool update_input = false;
    bool update_lut = false;

    PyObject* tile_coord;
    while (status_controller.running() && strand.pop(tile_coord)) {
        GridVector grid = nine_grid(tile_coord, tiles);
        auto result = op(bucket, update_input, update_lut, grid);
        update_input = result.first;

        // Add morphed tile unless it is completely transparent
        bool empty_result = result.second == ConstTiles::ALPHA_TRANSPARENT();
        bool full_result = result.second == ConstTiles::ALPHA_OPAQUE();

        // A constant tile being returned implies no morph was performed,
        // hence the lookup table must be fully populated for the next tile.
        update_lut = !(empty_result || full_result);

        // Only add non-transparent tiles to result, and only transfer ownership
        // for non-constant tiles (don't decref const tiles out of existence)
        if (!empty_result) morphed.set(tile_coord, result.second, !full_result);
    }
}

void
morph_worker(
    int offset, StrandQueue& queue, AtomicDict tiles,
    std::promise<AtomicDict> result, Controller& status_controller)
{
    AtomicDict morphed;
    Morpher bucket(abs(offset));
    Strand strand;
    while (status_controller.running() && queue.pop(strand)) {
        morph_strand(offset, strand, tiles, bucket, morphed, status_controller);
        status_controller.inc_processed(strand.size());
    }
    result.set_value(morphed);
}

// Entry point to morphological operations,
// this is what should be called from Python code.
void
morph(
    int offset, PyObject* morphed, PyObject* tiles, PyObject* strands,
    Controller& status_controller)
{
    if (offset == 0 || offset > N || offset < -N || !PyDict_Check(tiles) ||
        !PyList_CheckExact(strands)) {
        printf("Invalid morph parameters!\n");
        return;
    }
    const int min_strands_per_worker = 4;
    StrandQueue work_queue (strands);
    process_strands(
        morph_worker, offset, min_strands_per_worker, work_queue,
        AtomicDict(tiles), AtomicDict(morphed), status_controller);
}

bool
Morpher::input_fully_opaque()
{
    return all_equal_to<chan_t>(input, 2 * radius + N, fix15_one);
}

bool
Morpher::input_fully_transparent()
{
    return all_equal_to<chan_t>(input, 2 * radius + N, 0);
}
