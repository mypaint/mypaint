// hack needed as long as pygtk returns an old Numeric array instead of a NumPy one
// http://bugzilla.gnome.org/show_bug.cgi?id=397544
#include "Python.h"

typedef struct {
  PyObject_HEAD
  char *data;
  int nd;
  int *dimensions, *strides;
  PyObject *base;
  void *descr;
  int flags;
  PyObject *weakreflist;
} OldNumeric_PyArrayObject;

PyObject * gdkpixbuf_numeric2numpy(PyObject * gdk_pixbuf_pixels_array)
{
  // in case the bug is fixed in pygtk
  if (PyArray_Check(gdk_pixbuf_pixels_array)) {
    Py_INCREF(gdk_pixbuf_pixels_array);
    return gdk_pixbuf_pixels_array;
  }
  // we should type-check our argument, but I guess the assertions below will suffice

  OldNumeric_PyArrayObject * arr = (OldNumeric_PyArrayObject*)gdk_pixbuf_pixels_array;
  assert(arr->nd == 3);

  npy_intp dims[3];
  dims[0] = arr->dimensions[0];
  dims[1] = arr->dimensions[1];
  dims[2] = arr->dimensions[2];

  PyArrayObject * result = (PyArrayObject *)PyArray_SimpleNewFromData(arr->nd,
                                                                      dims, 
                                                                      PyArray_UBYTE,
                                                                      arr->data);

  if (result == NULL) return NULL;

  // pygtk sets only strides[0]
  if (result->strides[0] != arr->strides[0]) {
    result->strides[0] = arr->strides[0];
    // note: http://bugzilla.gnome.org/show_bug.cgi?id=447388
    result->flags &= ~NPY_CONTIGUOUS;
  }

  Py_INCREF(gdk_pixbuf_pixels_array);
  result->base = gdk_pixbuf_pixels_array;

  return PyArray_Return(result);
}
