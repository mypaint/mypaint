/* This file is part of MyPaint.
 * Copyright (C) 2019 by the MyPaint Development Team.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */
#ifndef FILL_COMMON_HPP
#define FILL_COMMON_HPP

#include "fix15.hpp"

#include <mypaint-config.h>

#include <Python.h>
#include <vector>

#include "common.hpp"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include <numpy/ndarraytypes.h>

#define N MYPAINT_TILE_SIZE

#ifndef MIN
#define MIN(a, b) (a) < (b) ? (a) : (b)
#endif

#ifndef MAX
#define MAX(a, b) (a) > (b) ? (a) : (b)
#endif

#define DFF(a, b) (a) > (b) ? ((a) - (b)) : ((b) - (a))

// channel type: chan_t, used for color and alpha channels
typedef fix15_short_t chan_t;

class ConstTiles
{
  public:
    static PyObject* ALPHA_OPAQUE();
    static PyObject* ALPHA_TRANSPARENT();

  private:
    static void init();
    static PyObject* _ALPHA_OPAQUE;
    static PyObject* _ALPHA_TRANSPARENT;
};

#ifdef SWIG
%ignore coord;
%ignore rgba;
%ignore PixelRef;
%ignore PixelBuffer;
%ignore GridVector;
%ignore init_from_nine_grid;
#endif

struct coord {
    coord(){};
    coord(int x, int y) : x(x), y(y) {}
    int x;
    int y;
};

/*
  Convenience struct corresponding to a chan_t[4],
  but with nicer creation and accessors
*/
struct rgba {
    chan_t red;
    chan_t green;
    chan_t blue;
    chan_t alpha;

    rgba() : red(0), green(0), blue(0), alpha(0) {}
    rgba(const rgba& v)
        : red(v.red), green(v.green), blue(v.blue), alpha(v.alpha)
    {
    }
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
        chan_t dr = DFF(red, b.red);
        chan_t db = DFF(blue, b.blue);
        chan_t dg = DFF(green, b.green);
        return MAX(dr, MAX(db, MAX(dg, DFF(alpha, b.alpha))));
    }

    inline bool operator!=(const rgba& b) const
    {
        return red != b.red || green != b.green || blue != b.blue ||
               alpha != b.alpha;
    }

    inline bool operator==(const rgba& b) const
    {
        return red == b.red && green == b.green && blue == b.blue &&
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
    PixelRef(C* pixel, const int x_stride, const int y_stride)
        : x_stride(x_stride), y_stride(y_stride), pixel(pixel)
    {
    }
    const C read() { return *pixel; }
    void write(C val) { *pixel = val; }
    inline void move_x(int dist) { pixel += dist * x_stride; }
    inline void move_y(int dist) { pixel += dist * y_stride; }
    inline C above() { return *(pixel - y_stride); }
    inline C below() { return *(pixel + y_stride); }
  private:
    const int x_stride;
    const int y_stride;
    C* pixel;
};

/*
  Wraps a PyArrayObject and provides some convenience accessors for
  improved code readability with minimal overhead
*/
template <typename C>
class PixelBuffer
{
  public:
    PyObject* array_ob;

    explicit PixelBuffer(PyObject* buf)
    {
        PyArrayObject* arr_buf = (PyArrayObject*)buf;
// Debug stuff
#ifdef HEAVY_DEBUG
        assert(PyArray_Check(buf));
        assert(PyArray_DIM(arr_buf, 0) == MYPAINT_TILE_SIZE);
        assert(PyArray_DIM(arr_buf, 1) == MYPAINT_TILE_SIZE);
        assert(PyArray_TYPE(arr_buf) == NPY_UINT16);
        assert(PyArray_DIM(arr_buf, 2) == sizeof(C));
        assert(PyArray_ISONESEGMENT(arr_buf));
        assert(PyArray_ISALIGNED(arr_buf));
        assert(PyArray_IS_C_CONTIGUOUS(arr_buf));
#endif
        this->array_ob = buf;
        this->x_stride = PyArray_STRIDE(arr_buf, 1) / sizeof(C);
        this->y_stride = PyArray_STRIDE(arr_buf, 0) / sizeof(C);
        this->buffer = reinterpret_cast<C*>(PyArray_BYTES(arr_buf));
    }
    static PixelBuffer<C> create_threadsafe(PyObject* buf)
    {
        PyGILState_STATE gstate;
        gstate = PyGILState_Ensure();
        auto pb = PixelBuffer(buf);
        PyGILState_Release(gstate);
        return pb;
    }
    PixelRef<C> get_pixel(unsigned int x, unsigned int y)
    {
        return PixelRef<C>(
            buffer + y * y_stride + x * x_stride, x_stride, y_stride);
    }
    bool is_uniform()
    {
        PixelRef<C> px = get_pixel(0, 0);
        C first = px.read();
        px.move_x(1);
        for (int i = 1; i < (N * N); i++, px.move_x(1))
            if (first != px.read()) return false;
        return true;
    }
    C& operator()(int x, int y)
    {
        return *(buffer + y * y_stride + x * x_stride);
    }

  private:
    int x_stride;
    int y_stride;
    C* buffer;
};

typedef std::vector<PixelBuffer<chan_t>> GridVector;

/*
  GIL-safe queue for strands used by worker processes
*/
template <typename T>
class AtomicQueue
{
  public:
    explicit AtomicQueue() {}
    explicit AtomicQueue(PyObject* items)
    {
        PyGILState_STATE gstate;
        gstate = PyGILState_Ensure();
        index = 0;
        num_strands = PyList_GET_SIZE(items);
        this->items = items;
        PyGILState_Release(gstate);
    }
    // Prevent copy construction (all workers should share it)
    AtomicQueue(AtomicQueue&) = delete;
    bool pop(T& item)
    {
        PyGILState_STATE gstate;
        gstate = PyGILState_Ensure();
        if (index >= num_strands) {
            PyGILState_Release(gstate);
            return false;
        } else {
            item = T(PyList_GET_ITEM(items, index));
            ++index;
            PyGILState_Release(gstate);
            return true;
        }
    }

  private:
    PyObject* items;
    Py_ssize_t index;
    Py_ssize_t num_strands;
};

/*
  Create and return a N x N rgba tile based on an rgb color
  and a N x N tile of alpha values
*/
PyObject* fill_rgba(
    PyObject* src, double fill_r, double fill_g, double fill_b, int min_x,
    int min_y, int max_x, int max_y);

// Read sections from a nine-grid of tiles to a single array
// r*r and N*r / r*N rectangles are read from corner and side
// tiles respectively (where r = radius)
void init_from_nine_grid(
    int radius, chan_t** input, bool from_above, GridVector grid);

#endif //__HAVE_FILL_COMMON_HPP
