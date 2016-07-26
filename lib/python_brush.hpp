/* brushlib - The MyPaint Brush Library
 * Copyright (C) 2011 Martin Renold <martinxyz@gmx.ch>
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#include <mypaint-brush-settings.h>

class PythonBrush : public Brush {

public:
  // get states as numpy array
  PyObject * get_states_as_array ()
  {
    npy_intp dims = {MYPAINT_BRUSH_STATES_COUNT};
    PyObject * data = PyArray_SimpleNew(1, &dims, NPY_FLOAT32);
    npy_float32 * data_p = (npy_float32*)PyArray_DATA((PyArrayObject*)data);
    for (int i=0; i<MYPAINT_BRUSH_STATES_COUNT; i++) {
      data_p[i] = get_state((MyPaintBrushState)i);
    }
    return data;
  }

  // set states from numpy array
  void set_states_from_array (PyObject * obj)
  {
    PyArrayObject* data = (PyArrayObject*)obj;
    assert(PyArray_NDIM(data) == 1);
    assert(PyArray_DIM(data, 0) == MYPAINT_BRUSH_STATES_COUNT);
    assert(PyArray_ISCARRAY(data));
    npy_float32 * data_p = (npy_float32*)PyArray_DATA(data);
    for (int i=0; i<MYPAINT_BRUSH_STATES_COUNT; i++) {
      set_state((MyPaintBrushState)i, data_p[i]);
    }
  }

  // Same as Brush::stroke_to() but with minimal exception handling:
  // don't indicate that a split is pending should an exception happen
  // in the surface code (e.g. out-of-memory)
  bool stroke_to (Surface * surface, float x, float y, float pressure, float xtilt, float ytilt, double dtime, float viewzoom, float viewrotation, float barrel_rotation)
  {
    bool res = Brush::stroke_to (surface, x, y, pressure, xtilt, ytilt, dtime, viewzoom, viewrotation, barrel_rotation);
    if (PyErr_Occurred()) {
      res = false;
    }
    return res;
  }

};
