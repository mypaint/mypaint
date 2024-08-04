/* This file is part of MyPaint.
 * Copyright (C) 2018 by the MyPaint Development Team.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */

#include "gap_detection.hpp"

DistanceBucket::DistanceBucket(int distance) : distance(distance)
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

static inline void
upd_dist(coord lc, PixelBuffer<chan_t>& dists, int new_dst)
{
    if (lc.x < 0 || lc.x > (N - 1) || lc.y < 0 || lc.y > (N - 1)) return;
    int curr_dist = dists(lc.x, lc.y);
    if (curr_dist > new_dst) {
        dists(lc.x, lc.y) = new_dst;
    }
}

using rot_op = std::function<coord(int x, int y, int x_offset, int y_offset)>;

// Search an octant with a radius of _dist_ pixels, marking any gaps
// that are found. The octant searched is determined by the rotation
// function provided.
bool
dist_search(
    int x, int y, int dist, chan_t** alphas, PixelBuffer<chan_t>& dists,
    rot_op op)
{

    // int d_lim = 1 + (dist * dist);
    int offs = dist + 1;
    int rx = x - offs;
    int ry = y - offs;

    coord t1 = op(x, y, 0, -1);
    coord t2 = op(x, y, 1, -1);

    if (alphas[t1.y][t1.x] == 0 || alphas[t2.y][t2.x] == 0) return false;

    bool gap_found = false;

    for (int yoffs = 2; yoffs < dist + 2; ++yoffs) {
        int y_dst_sqr = (yoffs - 1) * (yoffs - 1);

        for (int xoffs = 0; xoffs <= yoffs; ++xoffs) {
            int offs_dst = y_dst_sqr + (xoffs) * (xoffs);
            if (offs_dst >= 1 + dist * dist) break;
            coord c = op(x, y, xoffs, -yoffs);
            if (alphas[c.y][c.x] == 0) {

                // Gap found
                gap_found = true;

                // Double-width distance assignment
                float dx = (float)xoffs / (yoffs - 1);
                float tx = 0;
                int cx = 0;
                for (int cy = 1; cy < yoffs; ++cy) {
                    upd_dist(op(rx, ry, cx, 0 - cy), dists, offs_dst);
                    tx += dx;
                    if (floor(tx) > cx) {
                        cx++;
                        upd_dist(op(rx, ry, cx, 0 - cy), dists, offs_dst);
                    }
                    upd_dist(op(rx, ry, cx + 1, 0 - cy), dists, offs_dst);
                }
            }
        }
    }
    return gap_found;
}

// Coordinate reflection/rotation
coord
top_right(int x, int y, int x_offset, int y_offset)
{
    return coord(x + x_offset, y + y_offset);
}
coord
top_centr(int x, int y, int x_offset, int y_offset)
{
    return coord(x - y_offset, y - x_offset);
}
coord
bot_centr(int x, int y, int x_offset, int y_offset)
{
    return coord(x - y_offset, y + x_offset);
}
coord
bot_right(int x, int y, int x_offset, int y_offset)
{
    return coord(x + x_offset, y - y_offset);
}

/* Search for gaps in the 9-grid of flooded alpha tiles,
   a gap being defined as a
 */
bool
find_gaps(
    DistanceBucket& rb, PyObject* radiuses_arr, PyObject* mid, PyObject* n,
    PyObject* e, PyObject* s, PyObject* w, PyObject* ne, PyObject* se,
    PyObject* sw, PyObject* nw)
{
    int r = rb.distance + 1;

    typedef PixelBuffer<chan_t> PBT;
    GridVector input{PBT(nw), PBT(n),  PBT(ne), PBT(w), PBT(mid),
                     PBT(e),  PBT(se), PBT(s),  PBT(sw)};

    init_from_nine_grid(r, rb.input, false, input);

    bool gaps_found = false;

    PixelBuffer<chan_t> radiuses(radiuses_arr);
    // search for gaps in an approximate semi-circle
    for (int y = 0; y < 2 * r + N - 1;
         ++y) { // we check at most distance+1 pixels above any point
        for (int x = 0; x < r + N - 1; ++x) {
            if (rb.input[y][x] ==
                0) { // Search for gaps in relation to this pixel
                if (y >= r) {
                    gaps_found |= dist_search(
                        x, y, rb.distance, rb.input, radiuses, top_right);
                    gaps_found |= dist_search(
                        x, y, rb.distance, rb.input, radiuses, top_centr);
                }
                if (y < N + r) {
                    gaps_found |= dist_search(
                        x, y, rb.distance, rb.input, radiuses, bot_centr);
                    gaps_found |= dist_search(
                        x, y, rb.distance, rb.input, radiuses, bot_right);
                }
            }
        }
    }
    return gaps_found;
}
