/* This file is part of MyPaint.
 * Copyright (C) 2018 by the MyPaint Development Team.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */

#ifndef FLOODFILL_HPP
#define FLOODFILL_HPP

#include <queue>
#include <vector>

#include "common.hpp"

#include <Python.h>

#include "fix15.hpp"

#include <mypaint-config.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

#define N MYPAINT_TILE_SIZE
#define TILE_SQUARED N * N

// Largest squared gap distance - represents infinite radius
#define MAX_GAP 2*N*N

#ifndef MIN
#define MIN(a,b) (a) < (b) ? (a) : (b)
#endif

#ifndef MAX
#define MAX(a,b) (a) > (b) ? (a) : (b)
#endif

#define DFF(a,b) (a) > (b) ? ((a)-(b)) : ((b)-(a))

typedef fix15_short_t chan_t;


struct edges {
    enum edge {
        north = 0,
        east = 1,
        south = 2,
        west = 3,
        none = 4
    };
};
typedef edges::edge edge;


// These structures should not be used from Python code
#ifdef SWIG
%ignore coord;
%ignore rgba;
%ignore PixelRef;
%ignore PixelBuffer;
#endif


struct coord {
    coord() { };
    coord(int x, int y) : x(x), y(y) {}
    int x;
    int y;
};


/*
  Convenience struct corresponding to a chan_t[4],
  but with nicer creation and accessors
*/
struct rgba
{
    chan_t red;
    chan_t green;
    chan_t blue;
    chan_t alpha;

    rgba() : red (0), green(0), blue(0), alpha(0) {}
    rgba(const rgba &v) :
        red (v.red), green (v.green), blue (v.blue), alpha (v.alpha) {}
    rgba(double r, double g, double b, chan_t a)
        {
            red = fix15_short_clamp(r * a);
            green = fix15_short_clamp(g * a);
            blue = fix15_short_clamp(b * a);
            alpha = a;
        };
    rgba(chan_t r, chan_t g, chan_t b, chan_t a)
        {
            red = r;
            green = g;
            blue = b;
            alpha = a;
        };

    inline fix15_t max_diff(const rgba& b) const
        {
            return
                MAX(DFF(red, b.red),
                MAX(DFF(blue, b.blue),
                MAX(DFF(green, b.green),
                DFF(alpha, b.alpha))));
        }

    inline bool operator!=(const rgba& b) const
        {
            return
                red != b.red ||
                green != b.green ||
                blue != b.blue ||
                alpha != b.alpha;
        }
    inline bool operator==(const rgba& b) const
        {
            return
                red == b.red &&
                green == b.green &&
                blue == b.blue &&
                alpha == b.alpha;
        }
};


/*
  Abstracts a mutable reference to a pixel in a tile,
  hiding useful pointer arithmetic
*/
template <typename C>
class PixelRef
{
public:
    PixelRef(C *pixel, const int x_stride, const int y_stride) :
        x_stride (x_stride), y_stride (y_stride), pixel (pixel) {}
    const C& read() { return *pixel; }
    void write(C val) { *pixel = val; }
    inline void move_x(int dist) { pixel += dist * x_stride; }
    inline void move_y(int dist) { pixel += dist * y_stride; }
    inline C& above() { return *(pixel - y_stride); }
    inline C& below() { return *(pixel + y_stride); }
private:
    const int x_stride;
    const int y_stride;
    C *pixel;
};


/*
  Wraps a PyArrayObject and provides some convenience accessors for
  improved code readability with minimal overhead
*/
template <typename C>
class PixelBuffer
{
public:
    explicit PixelBuffer(PyArrayObject *buf) :
        x_stride (PyArray_STRIDE(buf, 1) / sizeof(C)),
        y_stride (PyArray_STRIDE(buf, 0) / sizeof(C)),
        buffer (reinterpret_cast<C*>(PyArray_BYTES(buf))) {}
    explicit PixelBuffer(PyObject *buf) :
        x_stride (PyArray_STRIDE((PyArrayObject*)buf, 1) / sizeof(C)),
        y_stride (PyArray_STRIDE((PyArrayObject*)buf, 0) / sizeof(C)),
        buffer (reinterpret_cast<C*>(PyArray_BYTES((PyArrayObject*)buf)))
        {}
    PixelRef<C>
    get_pixel(unsigned int x, unsigned int y)
        {
            return PixelRef<C>(buffer +
                                  y * y_stride +
                                  x * x_stride,
                                  x_stride, y_stride);
        }
    bool is_uniform()
        {
            PixelRef<C> px = get_pixel(0,0);
            C first = px.read();
            px.move_x(1);
            for(int i = 1; i < TILE_SQUARED; i++, px.move_x(1))
                if(first != px.read())
                    return false;
            return true;
        }
    C& operator()(int x, int y)
        {
            return *(buffer + y * y_stride + x * x_stride);
        }
private:
    const int x_stride;
    const int y_stride;
    C *buffer;
};


/*
  Implements the pixel threshold test function and uses it in the
  fill, alpha flooding, and tile uniformity/fillability methods
*/
class Filler
{
private:
    const rgba targ;
    const rgba targ_premult;
    const fix15_t tolerance;
    std::queue<coord> queue;
public:
    Filler(int targ_r, int targ_g, int targ_b, int targ_a, double tol);
    // Perform a scanline fill based on the rgba src tile and seed coordinates,
    // writing the resulting fill alphas to the dst tile and returning
    // any new overflows
    PyObject *
    fill(
        PyObject *src,
        PyObject *dst,
        PyObject *seeds,
        edge direction,
        int min_x, int min_y, int max_x, int max_y);
    // Flood the dst alpha tile with the fill alpha values
    // for the entire rgba src tile
    void
    flood(
        PyObject *src,
        PyObject *dst);
    // Test if the given rgba src tile is uniform
    // (all pixels having the same rgba color)), and if it is
    // returning the fill alpha for that color, otherwise Py_None
    PyObject *
    tile_uniformity(
        bool is_empty,
        PyObject *src);
protected:
    // Pixel threshold test - the return value indicates the alpha value of the
    // filled pixel, where an alpha of 0 indicates that the pixel should not be
    // filled (and the fill not propagated through that pixel).
    chan_t pixel_fill_alpha(const rgba &src_px);
    bool check_enqueue(const int x, const int y, bool check,
                       const rgba &src_px,
                       const chan_t &dst_px);
};


// A GapClosingFiller uses additional distance data
// to stop filling when leaving a detected gap
class GapClosingFiller
{
public:
    GapClosingFiller(int max_dist, bool track_seep);
    PyObject *
    fill(PyObject *alphas,
         PyObject *distances,
         PyObject *dst,
         PyObject *seeds,
         int min_x, int min_y,
         int max_x, int max_y);
    PyObject *
    unseep(PyObject* distances,
           PyObject *dst,
           PyObject *seeds,
           bool initial);
protected:
    const int max_distance;
    const bool track_seep;
};


/*
  Create and return a N x N rgba tile based on an rgb color
  and a N x N tile of alpha values
*/
PyObject* fill_rgba(
    PyObject *src, double fill_r, double fill_g, double fill_b,
    int min_x, int min_y, int max_x, int max_y);

#endif //__HAVE_FLOODFILL_HPP
