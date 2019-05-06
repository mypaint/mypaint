/* This file is part of MyPaint.
 * Copyright (C) 2019 by the MyPaint Development Team.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */

#include "blur.hpp"
#include "fill_constants.hpp"

#include <cmath>

// Generate gaussian multiplicands used for blurring.
// They are stored and used with fixed-point arithmetic
static const std::vector<fix15_short_t>
blur_factors(int r)
{
    constexpr double pi = 3.141592653589793;

    // Equations nicked from Krita
    float sigma = 0.3 * r + 0.3;
    int prelim_size = 6 * std::ceil(sigma + 1);
    float mul = 1 / sqrt(2 * pi * sigma * sigma);
    float exp_mul = 1 / (2 * sigma * sigma);

    std::vector<fix15_short_t> factors;
    int center = (prelim_size - 1) / 2;
    for (int i = 0; i < prelim_size; ++i) {
        int d = center - i;
        double fac = mul * exp(-d * d * exp_mul);
        // The bit-or'ing is a hack to avoid the sum of
        // multiplicands being less than 0, blurred pixels
        // are clamped to fix15_one anyway.
        factors.push_back((fix15_t)(fix15_one * fac) | 3);
    }
    return factors;
}

// Allocate memory for input and intermediate buffers
BlurBucket::BlurBucket(int r)
    : factors(blur_factors(r)), radius((factors.size() - 1) / 2)
{
    // Suppress uninitialization warning, the output
    // array is always fully populated before use
    const int d = N + radius * 2;
    // Output from 3x3-grid,
    // input to horizontal blur (Y x X) = (d x d)
    input_full = new chan_t*[d];
    for (int i = 0; i < d; ++i) {
        input_full[i] = new chan_t[d];
    }
    // Output for horizontal blur,
    // input to vertical blur (Y x X) = (d x N)
    input_vert = new chan_t*[d];
    for (int i = 0; i < d; ++i) {
        input_vert[i] = new chan_t[N];
    }
}

BlurBucket::~BlurBucket()
{
    const int d = N + radius * 2;
    for (int i = 0; i < d; ++i) {
        delete[] input_full[i];
        delete[] input_vert[i];
    }
    delete[] input_full;
    delete[] input_vert;
}

PyObject*
BlurBucket::blur(bool can_update, GridVector input_grid)
{
    initiate(can_update, input_grid);

    if (input_fully_opaque()) return ConstTiles::ALPHA_OPAQUE();

    if (input_fully_transparent()) return ConstTiles::ALPHA_TRANSPARENT();

    int r = radius;

    // Create output buffer
    npy_intp dims[] = {N, N};
    PyGILState_STATE gstate = PyGILState_Ensure();
    PyObject* out_tile = PyArray_EMPTY(2, dims, NPY_USHORT, 0);
    PixelBuffer<chan_t> out_buf(out_tile);
    PyGILState_Release(gstate);

    // Blur each row from input to intermediate buffer
    for (int y = 0; y < N + 2 * r; ++y) {
        for (int x = 0; x < N; ++x) {
            fix15_t blurred = 0;
            for (int xoffs = -r; xoffs < r + 1; xoffs++) {
                fix15_t in = input_full[y][x + xoffs + r];
                blurred += fix15_mul(in, factors[xoffs + r]);
            }
            input_vert[y][x] = fix15_short_clamp(blurred);
        }
    }

    // Blur each column from intermediate to output buffer
    for (int x = 0; x < N; ++x) {
        for (int y = 0; y < N; ++y) {
            fix15_t blurred = 0;
            for (int yoffs = -r; yoffs < r + 1; yoffs++) {
                fix15_t in = input_vert[y + yoffs + r][x];
                blurred += fix15_mul(in, factors[yoffs + r]);
            }
            out_buf(x, y) = fix15_short_clamp(blurred);
        }
    }

    return out_tile;
}

void
BlurBucket::initiate(bool can_update, GridVector input)
{
    init_from_nine_grid(radius, input_full, can_update, input);
}


bool
BlurBucket::input_fully_opaque()
{
    return all_eq<chan_t>(input_full, 2 * radius + N, fix15_one);
}

bool
BlurBucket::input_fully_transparent()
{
    return all_eq<chan_t>(input_full, 2 * radius + N, 0);
}

/*
  Blur a strand of tiles, from top to bottom. This function
  is very similar to morph_strand, but does not need to keep
  track of separate update flags. In fact, since the blur
  function always writes its input array before checking if it
  can skip the actual op, subsequent blurs can always update
  the input array.
*/
void
blur_strand(
    Strand& strand, AtomicDict& tiles, BlurBucket& bucket, AtomicDict& blurred)
{
    bool can_update = false;
    PyObject* tile_coord;
    while (strand.pop(tile_coord)) {
        GridVector grid = nine_grid(tile_coord, tiles);

        PyObject* result = bucket.blur(can_update, grid);
        can_update = true;

        // Add morphed tile unless it is completely transparent
        bool is_empty = result == ConstTiles::ALPHA_TRANSPARENT();
        bool is_full = result == ConstTiles::ALPHA_OPAQUE();
        if (!is_empty) blurred.set(tile_coord, result, !is_full);
    }
}

void
blur_worker(
    int radius, StrandQueue& queue, AtomicDict tiles,
    std::promise<AtomicDict> result)
{
    AtomicDict blurred;
    BlurBucket bucket(radius);
    Strand strand;
    while (queue.pop(strand)) {
        blur_strand(strand, tiles, bucket, blurred);
    }
    result.set_value(blurred);
}

void
blur(int radius, PyObject* blurred, PyObject* tiles, PyObject* strands)
{
    if (radius <= 0 || !PyDict_Check(tiles) || !PyList_CheckExact(strands)) {
        printf("Invalid blur parameters!\n");
        return;
    }

    const int min_strands_per_worker = 2;
    StrandQueue work_queue(strands);
    process_strands(
        blur_worker, radius, min_strands_per_worker, std::ref(work_queue),
        AtomicDict(tiles), AtomicDict(blurred));
}
