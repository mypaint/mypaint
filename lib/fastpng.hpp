#define PNG_SKIP_SETJMP_CHECK
#include "png.h"

#ifndef SWIG
static void png_write_error_callback(png_structp png_save_ptr, png_const_charp error_msg)
{
  // we don't trust libpng to call the error callback only once, so
  // check for already-set error
  if (!PyErr_Occurred()) {
    if (!strcmp(error_msg, "Write Error")) {
      PyErr_SetFromErrno(PyExc_IOError);
    } else {
      PyErr_Format(PyExc_RuntimeError, "Error writing PNG: %s", error_msg);
    }
  }
  longjmp (png_jmpbuf(png_save_ptr), 1);
}
#endif

PyObject * save_png_fast_progressive(char * filename, int w, int h, bool has_alpha,
                                     PyObject * get_data_callback)
{
  png_structp png_ptr = NULL;
  png_infop info_ptr = NULL;
  PyObject * result = NULL;
  int bpc;
  FILE * fp = NULL;

  /* TODO: try if this silliness helps
#if defined(PNG_LIBPNG_VER) && (PNG_LIBPNG_VER >= 10200)
  png_uint_32 mask, flags;
  
  flags = png_get_asm_flags(png_ptr);
  mask = png_get_asm_flagmask(PNG_SELECT_READ | PNG_SELECT_WRITE);
  png_set_asm_flags(png_ptr, flags | mask);
#endif
  */

  bpc = 8;
  
  fp = fopen(filename, "wb");
  if (!fp) {
    PyErr_SetFromErrno(PyExc_IOError);
    //PyErr_Format(PyExc_IOError, "Could not open PNG file for writing: %s", filename);
    goto cleanup;
  }

  png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, (png_voidp)NULL, png_write_error_callback, NULL);
  if (!png_ptr) {
    PyErr_SetString(PyExc_MemoryError, "png_create_write_struct() failed");
    goto cleanup;
  }
  
  info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr) {
    PyErr_SetString(PyExc_MemoryError, "png_create_info_struct() failed");
    goto cleanup;
  }

  if (setjmp(png_jmpbuf(png_ptr))) {
    goto cleanup;
  }

  png_init_io(png_ptr, fp);

  png_set_IHDR (png_ptr, info_ptr,
                w, h, bpc,
                has_alpha ? PNG_COLOR_TYPE_RGB_ALPHA : PNG_COLOR_TYPE_RGB,
                PNG_INTERLACE_NONE,
                PNG_COMPRESSION_TYPE_BASE,
                PNG_FILTER_TYPE_BASE);

  // default (all filters enabled):                 1350ms, 3.4MB
  //png_set_filter(png_ptr, 0, PNG_FILTER_NONE);  // 790ms, 3.8MB
  //png_set_filter(png_ptr, 0, PNG_FILTER_PAETH); // 980ms, 3.5MB
  png_set_filter(png_ptr, 0, PNG_FILTER_SUB);     // 760ms, 3.4MB

  //png_set_compression_level(png_ptr, 0); // 0.49s, 32MB
  //png_set_compression_level(png_ptr, 1); // 0.98s, 9.6MB
  png_set_compression_level(png_ptr, 2);   // 1.08s, 9.4MB
  //png_set_compression_level(png_ptr, 9); // 18.6s, 9.3MB

  png_write_info(png_ptr, info_ptr);

  {
    int y = 0;
    while (y < h) {
      PyObject * arr;
      int rows;
      arr = PyObject_CallObject(get_data_callback, NULL);

      if (!arr) goto cleanup;
#ifdef HEAVY_DEBUG
      assert(PyArray_ISCARRAY(arr));
      assert(PyArray_NDIM(arr) == 3);
      assert(PyArray_DIM(arr, 1) == w);
      assert(PyArray_DIM(arr, 2) == has_alpha?4:3);
      assert(PyArray_TYPE(arr) == NPY_UINT8);
#endif

      rows = PyArray_DIM(arr, 0);
      assert(rows > 0);
      y += rows;
      png_bytep p = (png_bytep)PyArray_DATA(arr);
      for (int row=0; row<rows; row++) {
        png_write_row (png_ptr, p);
        p += w * PyArray_DIM(arr, 2);
      }
      Py_DECREF(arr);
    }
    assert(y == h);
  }
  
  png_write_end (png_ptr, info_ptr);

  Py_INCREF(Py_None);
  result = Py_None;

 cleanup:
  if (info_ptr) png_destroy_write_struct (&png_ptr, &info_ptr);
  if (fp) fclose(fp);
  return result;
}
