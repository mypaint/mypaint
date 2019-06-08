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

#include "../fix15.hpp"

#include <mypaint-config.h>

#include <Python.h>
#include <vector>
#include <future>
#include <mutex>

#include "../common.hpp"

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

// Coordinate
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
  hiding the pointer arithmetic.
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
  improved code readability with minimal overhead.
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

/*
  Create a new alpha tile and return it wrapped in a PixelBuffer.
  This function is GIL-threadsafe.
  NOTE: This allocates a new numpy array which requires manual
  reference counting.
*/
PixelBuffer<chan_t> new_alpha_tile();

/*
  GIL-safe queue for strands used by worker processes

  This structure wraps a python list and provides an
  access-only interface that can be called from multiple
  threads.

  WARNING: The list reference is borrowed, not owned!
  It is up to the user to ensure that the list is not garbage
  collected during the lifetime of any wrapper using it.
*/
template <typename T>
class AtomicQueue
{
  public:
    // Create an empty queue (for declarations)
    explicit AtomicQueue() {}
    // Create a queue from a pointer to a PyList.
    explicit AtomicQueue(PyObject* items)
    {
        PyGILState_STATE gstate = PyGILState_Ensure();
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
    // Get the size of the queue
    Py_ssize_t size() { return num_strands; }

  private:
    PyObject* items;
    Py_ssize_t index;
    Py_ssize_t num_strands;
};

/*
  A strand is a sequence of vertically contiguous tile coordinates
  e.g. {(3, 4), (3, 5), (3, 6)}
*/
using Strand = AtomicQueue<PyObject*>;
using StrandQueue = AtomicQueue<Strand>;

/*
  GIL-threadsafe PyDict wrapper

  Note: Ownership of the PyDict IS handled by the wrapper,
  ensuring its validity as long as reference counting is handled
  correctly by other users of the PyDict.
*/
class AtomicDict
{
  public:
    // Create a new PyDict dictionary
    AtomicDict();
    // Borrow an existing PyDict
    explicit AtomicDict(PyObject* d);
    // Custom copy constructor for reference consistency
    AtomicDict(const AtomicDict&);
    // Relinquish ownership of wrapped dictionary
    ~AtomicDict();
    // Get an item from the dictionary
    PyObject* get(PyObject* key);
    // Add an item to the dictionary associated to the given key,
    // overriding existing items, if the key already exists
    // If transfer_ownership is true, the input's refcount is decremented
    void set(PyObject* key, PyObject* item, bool transfer_ownership = false);
    // Merge another dictionary into this one, overriding items
    // in this dictionary if the keys exist in both.
    void merge(AtomicDict&);
    // Get the size of the dictionary
    Py_ssize_t size();

  private:
    PyObject* dict;
};

typedef std::vector<PixelBuffer<chan_t>> GridVector;

/*
  For the given tile coordinate, return a vector of pixel buffers for
  the tiles of the coordinate and its 8 neighbours. If a neighbouring
  tile does not exist in the given tile dictionary, the constant empty
  alpha tile takes its place.

  Order of tiles in vector, where 4 is the tile of the input coordinate:
  0 1 2
  3 4 5
  6 7 8
*/
GridVector
nine_grid(PyObject* tile_coord, AtomicDict& tiles);

/*
   Read sections from a nine-grid of tiles to a single array
   r*r and N*r / r*N rectangles are read from corner and side
   tiles respectively (where r = radius)
*/
void init_from_nine_grid(
    int radius, chan_t** input, bool from_above, GridVector grid);

/*
  Helper function for checking if all items in
  a quadratic array equals the given value.
*/
template <typename T>
static bool
all_equal_to(T** arr, int dim, T val)
{
    for (int y = 0; y < dim; ++y)
        for (int x = 0; x < dim; ++x)
            if (arr[y][x] != val) return false;
    return true;
}


/*
  Used to track operation progress (such as number of tiles filled/morphed etc.)
  and permit cancellation of operations. Instances are shared between threads.
*/
class Controller {
public:
    Controller(Controller&) = delete;
    Controller() : run(true), tiles_processed(0) {}
    void stop() { run = false; }
    bool running() { return run; }
    void inc_processed(int incr)
    {
        inc_mutex.lock();
        tiles_processed += incr;
        inc_mutex.unlock();
    }
    int num_processed() { return tiles_processed; }
    void reset() { tiles_processed = 0; }
private:
    // Whether to keep running or not
    bool run;
    // The number of tiles processed for the current stage
    int tiles_processed;
    // tiles_processed incrementation mutex
    // GIL is not used as mutex here since Python API access is not required
    // and it is unnecessary to block this on activity in the GUI thread
    std::mutex inc_mutex;
};

/*
  Worker, processing strands of tiles from a distributed workload
*/
using worker_function = std::function<void(
    int offset, StrandQueue& input_strands, AtomicDict input_tiles,
    std::promise<AtomicDict> result, Controller& status_controller)>;

/*
  Return the recommended amount of worker threads, based on
  hardware capability and the size of the input (number of strands)
*/
int
num_strand_workers(int num_strands, int min_strands_per_worker);

/*
  Process a set of strands using a given worker function, potentially
  using multiple threads, and merge the result into the provided dictionary.
*/
void process_strands(
    worker_function worker, int offset, int min_strands_per_worker,
    StrandQueue& strands, AtomicDict tiles, AtomicDict result,
    Controller& status_controller);

#endif //FILL_COMMON_HPP
