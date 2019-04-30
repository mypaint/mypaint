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
#include <thread>
#include <future>
#include <tuple>

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
    for(size_t len_i = 1; len_i < se_lengths.size(); len_i++) {
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

void
MorphBucket::initiate(bool can_update, GridVector grid)
{
    init_from_nine_grid(radius, input, can_update, grid);
}

template <chan_t init, chan_t lim, op cmp>
static std::pair<bool, PixelBuffer<chan_t>>
generic_morph(
    MorphBucket &mb,
    bool can_update, GridVector input)
{
    PyGILState_STATE gstate;

    if (mb.can_skip<lim>(input[4])) {
        PyObject *skip_tile;
        if (lim == 0)
            skip_tile = ConstTiles::ALPHA_TRANSPARENT();
        else
            skip_tile = ConstTiles::ALPHA_OPAQUE();
        gstate = PyGILState_Ensure();
        auto skip_buf = PixelBuffer<chan_t>(skip_tile);
        PyGILState_Release(gstate);
        return std::make_pair(false, skip_buf);
    }

    mb.initiate(can_update, input);

    npy_intp dims[] = {N, N};

    gstate = PyGILState_Ensure();
    PixelBuffer<chan_t> dst_buf (PyArray_EMPTY(2, dims, NPY_USHORT, 0));
    PyGILState_Release(gstate);

    mb.morph<init, lim, cmp>(
        can_update, dst_buf);

    return std::make_pair(true, dst_buf);
}

inline chan_t max(chan_t a, chan_t b)
{
    return a > b ? a : b;
}


inline chan_t min(chan_t a, chan_t b)
{
    return a < b ? a : b;
}

std::pair<bool, PixelBuffer<chan_t>>
dilate(
    MorphBucket &mb,
    bool can_update, GridVector input)
{
    return generic_morph<0, fix15_one, max>(mb, can_update, input);
}

std::pair<bool, PixelBuffer<chan_t>>
erode(
    MorphBucket &mb,
    bool can_update, GridVector input)
{
    return generic_morph<fix15_one, 0, min>(mb, can_update, input);
}

// For the given tile coordinate, return a vector of pixel buffers for
// the tiles of the coordinate and its 8 neighbours. If a neighbouring
// tile does not exist, the empty alpha tile takes its place.
// Order of tiles in vector, where 4 is the input tile:
// 0 1 2
// 3 4 5
// 6 7 8
GridVector
nine_grid(PyObject *tile_coord, PyObject *tiles)
{
    const int num_tiles = 9;
    const int offs[] {-1, 0, 1};

    int x, y;

    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();

    PyArg_ParseTuple(tile_coord, "ii", &x, &y);
    std::vector<PixelBuffer<chan_t>> grid;

    for(int i = 0; i < num_tiles; ++i)
    {
        int _x = x + offs[i%3];
        int _y = y + offs[i/3];
        PyObject * c = Py_BuildValue("ii", _x, _y);
        PyObject *tile = PyDict_GetItem(tiles, c);
        Py_DECREF(c);
        if (tile)
            grid.push_back(PixelBuffer<chan_t>(tile));
        else
            grid.push_back(PixelBuffer<chan_t>(ConstTiles::ALPHA_TRANSPARENT()));
    }
    PyGILState_Release(gstate);

    return grid;
}

// Conditionally check if the alpha buffer output is completely transparent
bool
empty_result(int offset, PixelBuffer<chan_t> src_buf, PixelBuffer<chan_t> result_buf)
{
    auto t_tile = ConstTiles::ALPHA_TRANSPARENT();
    if(result_buf.array_ob == t_tile)
        return true;
    if(offset > 0 && src_buf.array_ob != t_tile)
        return false;
    else
        return result_buf(0,0) == 0 && result_buf.is_uniform();
}

// Morph a single strand of tiles, storing
// the output tiles in a Python dictionary "morphed"
void
morph_strand(
    int offset, // Dilation/erosion radius (+/-)
    Py_ssize_t strand_size,
    PyObject *strand,
    PyObject *tiles,
    MorphBucket &bucket,
    PyObject *morphed
    )
{
    auto op = offset > 0 ? dilate : erode;
    bool can_update = false;
    for(Py_ssize_t i = 0; i < strand_size; ++i)
    {
        PyGILState_STATE gstate;
        gstate = PyGILState_Ensure();
        PyObject *tile_coord = PyList_GET_ITEM(strand, i);
        PyGILState_Release(gstate);
        GridVector grid = nine_grid(tile_coord, tiles);
        auto result = op(
            bucket, can_update, grid);
        can_update = result.first;

        // Add morphed tile unless it is completely transparent
        if(!empty_result(offset, grid[4], result.second))
        {
            gstate = PyGILState_Ensure();
            PyDict_SetItem(morphed, tile_coord, result.second.array_ob);
            PyGILState_Release(gstate);
        }
    }
}

// Worker, processing strands of tiles from a distributed workload
void morph_worker(
    int offset, Py_ssize_t num_strands,
    PyObject *strands, PyObject *tiles,
    std::promise<PyObject*> result, int &index)
{
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();
    PyObject *morphed = PyDict_New();
    PyGILState_Release(gstate);
    MorphBucket bucket (abs(offset));
    while(true)
    {
        // Claim the GIL and check if there are more strands
        int i;
        gstate = PyGILState_Ensure();
        i = index++;
        if(i >= num_strands)
        {
            // No more strands, release the GIL and stop working
            PyGILState_Release(gstate);
            break;
        }
        // We're still the GIL holder, grab a strand and get its size
        PyObject *strand = PyList_GET_ITEM(strands, i);
        Py_ssize_t strand_size = PyList_GET_SIZE(strand);
        // We're done using the GIL
        PyGILState_Release(gstate);
        // Morph the strand, putting the result in the morphed dict
        morph_strand(offset, strand_size, strand, tiles, bucket, morphed);
    }
    // Job's done, return result to main thread
    result.set_value(morphed);
}

// Entry point to morphological operations,
// this is what should be called from Python code.
void
morph(int offset, PyObject *morphed, PyObject *tiles, PyObject *strands)
{
    if (offset == 0 || offset > N || offset < -N ||
        !PyDict_Check(tiles) || !PyList_CheckExact(strands))
    {
        printf("Invalid morph parameters!\n");
        return;
    }

    Py_ssize_t num_strands = PyList_GET_SIZE(strands);
    int max_threads = std::thread::hardware_concurrency();
    int max_by_strands = num_strands / 4;
    int num_threads = MIN(max_threads, max_by_strands);
    if(num_threads > 1)
    {
        std::vector<std::thread> threads (num_threads);
        std::vector<std::future<PyObject*>> futures (num_threads);

        PyEval_InitThreads();
        int strand_index = 0;

        // Create worker threads
        for(int i = 0; i < num_threads; ++i)
        {
            std::promise<PyObject*> promise;
            futures[i] = promise.get_future();
            threads[i] = std::thread(
                morph_worker,
                offset, num_strands, strands, tiles,
                std::move(promise),
                std::ref(strand_index)
                );
        }

        // Release the lock to let the workers work
        Py_BEGIN_ALLOW_THREADS

        for(int i = 0; i < num_threads; ++i)
        {
            // Wait for the output from the threads
            // and merge it into the final result
            futures[i].wait();
            PyObject *_m = futures[i].get();
            PyGILState_STATE state;
            state = PyGILState_Ensure();
            PyDict_Update(morphed, _m);
            Py_DECREF(_m);
            PyGILState_Release(state);
            threads[i].join();
        }

        // Reclaim the lock before returning to not make Python explode
        Py_END_ALLOW_THREADS
    }
    else
    {
        MorphBucket bucket (abs(offset));
        for (Py_ssize_t i = 0; i < num_strands; ++i)
        {
            PyObject *strand = PyList_GET_ITEM(strands, i);
            Py_ssize_t strand_size = PyList_GET_SIZE(strand);
            morph_strand(offset, strand_size, strand, tiles, bucket, morphed);
        }
    }
}


// Box blur parameters & memory allocation

// Generate gaussian multiplicands used for blurring.
// They are stored and used with fixed-point arithmetic
static const std::vector<fix15_short_t> blur_factors(int r)
{
    constexpr double pi = 3.141592653589793;

    // Equations nicked from Krita
    float sigma = 0.3 * r + 0.3;
    int prelim_size = 6 * std::ceil(sigma + 1);
    float mul = 1 / sqrt(2 * pi * sigma * sigma);
    float exp_mul = 1 / (2 * sigma * sigma);

    std::vector<fix15_short_t> factors;
    int center = (prelim_size - 1) / 2;
    for(int i = 0; i < prelim_size; ++i)
    {
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
BlurBucket::BlurBucket(int r) :
    factors (blur_factors(r)), radius ((factors.size()-1)/2)
{
    // Suppress uninitialization warning, the output
    // array is always fully populated before use
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

PyObject* BlurBucket::blur(bool can_update, GridVector input_grid)
{
    initiate(can_update, input_grid);

    if(input_fully_opaque())
        return ConstTiles::ALPHA_OPAQUE();

    if(input_fully_transparent())
        return ConstTiles::ALPHA_TRANSPARENT();

    int r = radius;

    // Blur each row from input to intermediate buffer
    for(int y = 0; y < N + 2 * r; ++y)
    {
        for(int x = 0; x < N; ++x)
        {
            fix15_t blurred = 0;
            for(int xoffs = -r; xoffs < r+1; xoffs++)
            {
                fix15_t in = input_full[y][x+xoffs + r];
                blurred += fix15_mul(in, factors[xoffs + r]);
            }
            input_vert[y][x] = fix15_short_clamp(blurred);
        }
    }

    // Blur each column from intermediate to output buffer
    for(int x = 0; x < N; ++x)
    {
        for(int y = 0; y < N; ++y)
        {
            fix15_t blurred = 0;
            for(int yoffs = -r; yoffs < r+1; yoffs++)
            {
                fix15_t in = input_vert[y+yoffs + r][x];
                blurred += fix15_mul(in, factors[yoffs + r]);
            }
            output[y][x] = fix15_short_clamp(blurred);
        }
    }

    npy_intp dims[] = {N, N};

    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();

    PyObject *out_tile = PyArray_EMPTY(2, dims, NPY_USHORT, 0);
    PixelBuffer<chan_t> out_buf (out_tile);

    PyGILState_Release(gstate);

    PixelRef<chan_t> out_px = out_buf.get_pixel(0,0);
    for(int y = 0; y < N; ++y) {
        for(int x = 0; x < N; ++x) {
            out_px.write(output[y][x]);
            out_px.move_x(1);
        }
    }

    return out_tile;
}

void BlurBucket::initiate(bool can_update, GridVector input)
{
    init_from_nine_grid(radius, input_full, can_update, input);
}

bool BlurBucket::input_fully_opaque()
{
    int dim = 2 * radius + N;
    for(int y = 0; y < dim; ++y)
        for(int x = 0; x < dim; ++x)
            if (input_full[y][x] != fix15_one)
                return false;
    return true;
}

bool BlurBucket::input_fully_transparent()
{
    int dim = 2 * radius + N;
    for(int y = 0; y < dim; ++y)
        for(int x = 0; x < dim; ++x)
            if (input_full[y][x] != 0)
                return false;
    return true;
}

void
blur_strand(
    Py_ssize_t strand_size,
    PyObject *strand,
    PyObject *tiles,
    BlurBucket &bucket,
    PyObject *blurred
    )
{
    bool can_update = false;
    for(Py_ssize_t i = 0; i < strand_size; ++i)
    {
        PyGILState_STATE gstate;
        gstate = PyGILState_Ensure();
        PyObject *tile_coord = PyList_GET_ITEM(strand, i);
        PyGILState_Release(gstate);
        GridVector grid = nine_grid(tile_coord, tiles);

        PyObject *result = bucket.blur(can_update, grid);
        can_update = true;

        // Add morphed tile unless it is completely transparent
        if(result != ConstTiles::ALPHA_TRANSPARENT())
        {
            gstate = PyGILState_Ensure();
            PyDict_SetItem(blurred, tile_coord, result);
            PyGILState_Release(gstate);
        }
    }
}

void
blur_worker(
    int radius, Py_ssize_t num_strands,
    PyObject *strands, PyObject *tiles,
    std::promise<PyObject*> result, int &index)
{
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();
    PyObject *blurred = PyDict_New();
    PyGILState_Release(gstate);
    BlurBucket bucket (radius);
    while(true)
    {
        // Claim the GIL and check if there are more strands
        int i;
        gstate = PyGILState_Ensure();
        i = index++;
        if(i >= num_strands)
        {
            // No more strands, release the GIL and stop working
            PyGILState_Release(gstate);
            break;
        }
        // We're still the GIL holder, grab a strand and get its size
        PyObject *strand = PyList_GET_ITEM(strands, i);
        Py_ssize_t strand_size = PyList_GET_SIZE(strand);
        // We're done using the GIL
        PyGILState_Release(gstate);
        // Morph the strand, putting the result in the morphed dict
        blur_strand(strand_size, strand, tiles, bucket, blurred);
    }
    // Job's done, return result to main thread
    result.set_value(blurred);
}


void
blur(int radius, PyObject *blurred, PyObject *tiles, PyObject *strands)
{
    if (radius <= 0 || !PyDict_Check(tiles) || !PyList_CheckExact(strands))
    {
        printf("Invalid blur parameters!\n");
        return;
    }

    Py_ssize_t num_strands = PyList_GET_SIZE(strands);
    int max_threads = std::thread::hardware_concurrency();
    int max_by_strands = num_strands / 2;
    int num_threads = MIN(max_threads, max_by_strands);
    if(num_threads > 1)
    {
        std::vector<std::thread> threads (num_threads);
        std::vector<std::future<PyObject*>> futures (num_threads);

        PyEval_InitThreads();
        int strand_index = 0;

        // Create worker threads
        for(int i = 0; i < num_threads; ++i)
        {
            std::promise<PyObject*> promise;
            futures[i] = promise.get_future();
            threads[i] = std::thread(
                blur_worker,
                radius, num_strands, strands, tiles,
                std::move(promise),
                std::ref(strand_index)
                );
        }

        // Release the lock to let the workers work
        Py_BEGIN_ALLOW_THREADS

        for(int i = 0; i < num_threads; ++i)
        {
            // Wait for the output from the threads
            // and merge it into the final result
            futures[i].wait();
            PyObject *_m = futures[i].get();
            PyGILState_STATE state;
            state = PyGILState_Ensure();
            PyDict_Update(blurred, _m);
            Py_DECREF(_m);
            PyGILState_Release(state);
            threads[i].join();
        }

        // Reclaim the lock before returning to not make Python explode
        Py_END_ALLOW_THREADS
    }
    else
    {
        BlurBucket bucket (radius);
        for (Py_ssize_t i = 0; i < num_strands; ++i)
        {
            PyObject *strand = PyList_GET_ITEM(strands, i);
            Py_ssize_t strand_size = PyList_GET_SIZE(strand);
            blur_strand(strand_size, strand, tiles, bucket, blurred);
        }
    }
}


// Gap closing stuff - prob move this out somewhere

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
bool dist_search(int x, int y, int dist,
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
        return false;

    bool gap_found = false;

    for(int yoffs = 2; yoffs < dist + 2; ++yoffs) {
        int y_dst_sqr = (yoffs-1)*(yoffs-1);

        for(int xoffs = 0; xoffs <= yoffs; ++xoffs) {
            int offs_dst = y_dst_sqr + (xoffs)*(xoffs);
            //int real_offs = (int) round(sqrtf((float) offs_dst));
            if (offs_dst >= 1 + dist * dist)
                break;
            coord c = op(x, y, xoffs, -yoffs);
            if(alphas[c.y][c.x] == 0) { // Gap found

                gap_found = true;

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
    return gap_found;
}

// Coordinate reflection/rotation
coord top_right(int x, int y, int xoffs, int yoffs) { return coord(x + xoffs, y + yoffs); }
coord top_centr(int x, int y, int xoffs, int yoffs) { return coord(x - yoffs, y - xoffs); }
coord bot_centr(int x, int y, int xoffs, int yoffs) { return coord(x - yoffs, y + xoffs); }
coord bot_right(int x, int y, int xoffs, int yoffs) { return coord(x + xoffs, y - yoffs); }

/* Search for gaps in the 9-grid of flooded alpha tiles,
   a gap being defined as a
 */
bool find_gaps(
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

    typedef PixelBuffer<chan_t> PBT;
    GridVector input {
        PBT(nw), PBT(n), PBT(ne),
        PBT(w), PBT(mid), PBT(e),
        PBT(se), PBT(s), PBT(sw)
    };

    init_from_nine_grid(r, rb.input, false, input);

    bool gaps_found = false;

    PixelBuffer<chan_t> radiuses (radiuses_arr);
    // search for gaps in an approximate semi-circle
    for (int y = 0; y < 2*r + N-1; ++y) { // we check at most distance+1 pixels above any point
        for (int x = 0; x < r + N-1; ++x) {
            if(rb.input[y][x] == 0) { // Search for gaps in relation to this pixel
                if (y >= r) {
                    gaps_found |= dist_search(x, y, rb.distance, rb.input, radiuses, top_right);
                    gaps_found |= dist_search(x, y, rb.distance, rb.input, radiuses, top_centr);
                }
                if (y < N + r) {
                    gaps_found |= dist_search(x, y, rb.distance, rb.input, radiuses, bot_centr);
                    gaps_found |= dist_search(x, y, rb.distance, rb.input, radiuses, bot_right);
                }
            }
        }
    }
    return gaps_found;
}
