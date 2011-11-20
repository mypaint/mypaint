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
  
  png_write_end (png_ptr, NULL);

  Py_INCREF(Py_None);
  result = Py_None;

 cleanup:
  if (info_ptr) png_destroy_write_struct (&png_ptr, &info_ptr);
  if (fp) fclose(fp);
  return result;
}

#ifndef SWIG
static void png_read_error_callback(png_structp png_read_ptr, png_const_charp error_msg)
{
  // we don't trust libpng to call the error callback only once, so
  // check for already-set error
  if (!PyErr_Occurred()) {
    if (!strcmp(error_msg, "Read Error")) {
      PyErr_SetFromErrno(PyExc_IOError);
    } else {
      PyErr_Format(PyExc_RuntimeError, "Error reading PNG: %s", error_msg);
    }
  }
  longjmp (png_jmpbuf(png_read_ptr), 1);
}
#endif

// Read a PNG progressively as 8bit RGBA. Signature of the callback:
//
// numpy_array = callback(full_image_width, full_image_height)
//
// The callback must return a writeable array of the image width.  If
// the height is smaller than the image height, the callback will be
// called again until the full image has been processed.
PyObject * load_png_fast_progressive(char * filename,
                                     PyObject * get_buffer_callback)
{
  // Note: we are not using the method that libpng calls "Reading PNG
  // files progressively". That method would involve feeding the data
  // into libpng piece by piece, which is not necessary if we can give
  // libpng a simple FILE pointer.

  png_structp png_ptr = NULL;
  png_infop info_ptr = NULL;
  PyObject * result = NULL;
  //int bpc;
  FILE * fp = NULL;
  int width, height;
  int rows_left;
  int color_type, bit_depth;

  fp = fopen(filename, "rb");
  if (!fp) {
    PyErr_SetFromErrno(PyExc_IOError);
    //PyErr_Format(PyExc_IOError, "Could not open PNG file for writing: %s", filename);
    goto cleanup;
  }

  png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, (png_voidp)NULL, png_read_error_callback, NULL);
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

  png_read_info(png_ptr, info_ptr);
  
  if (png_get_interlace_type(png_ptr, info_ptr) != PNG_INTERLACE_NONE) {
    PyErr_SetString(PyExc_RuntimeError, "Interlaced PNG files are not supported!");
  } 

  color_type = png_get_color_type(png_ptr, info_ptr);
  bit_depth = png_get_bit_depth(png_ptr, info_ptr);

  if (color_type == PNG_COLOR_TYPE_PALETTE) {
    png_set_palette_to_rgb(png_ptr);
  }
  
  if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) {
    png_set_expand_gray_1_2_4_to_8(png_ptr);
  }

  if (png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS)) {
    png_set_tRNS_to_alpha(png_ptr);
  }

  if (bit_depth == 16) png_set_strip_16(png_ptr);
  if (bit_depth < 8) png_set_packing(png_ptr);

  if (color_type == PNG_COLOR_TYPE_RGB ||
      color_type == PNG_COLOR_TYPE_GRAY) {
    png_set_add_alpha(png_ptr, 0xFF, PNG_FILLER_AFTER);
  }

  if (color_type == PNG_COLOR_TYPE_GRAY ||
      color_type == PNG_COLOR_TYPE_GRAY_ALPHA) {
    png_set_gray_to_rgb(png_ptr);
  }

  // TODO: do we need gamma transformation, or can we just assume
  //       sRGB in, sRGB out?

  png_read_update_info(png_ptr, info_ptr);

  // Verify what we have done
  if (png_get_bit_depth(png_ptr, info_ptr) != 8) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to convince libpng to convert to 8 bits per channel");
    goto cleanup;
  }
  if (png_get_color_type(png_ptr, info_ptr) != PNG_COLOR_TYPE_RGB_ALPHA) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to convince libpng to convert to RGBA (wrong color_type)");
    goto cleanup;
  }
  if (png_get_channels(png_ptr, info_ptr) != 4) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to convince libpng to convert to RGBA (wrong number of channels)");
    goto cleanup;
  }

  width = png_get_image_width(png_ptr, info_ptr);
  height = png_get_image_height(png_ptr, info_ptr);
  rows_left = height;
  
  while (rows_left) {
    PyObject * arr;
    int rows, row;
    png_bytep * row_pointers;
    
    arr = PyObject_CallFunction(get_buffer_callback, "ii", width, height);
    if (!arr) goto cleanup;
#ifdef HEAVY_DEBUG
    //assert(PyArray_ISCARRAY(arr));
    assert(PyArray_NDIM(arr) == 3);
    assert(PyArray_DIM(arr, 1) == width);
    assert(PyArray_DIM(arr, 2) == 4);
    assert(PyArray_TYPE(arr) == NPY_UINT8);
    assert(PyArray_ISBEHAVED(arr));
    assert(PyArray_STRIDE(arr, 1) == 4*sizeof(uint8_t));
    assert(PyArray_STRIDE(arr, 2) ==   sizeof(uint8_t));
#endif
    rows = PyArray_DIM(arr, 0);

    if (rows > rows_left) {
      PyErr_Format(PyExc_RuntimeError, "Attempt to read %d rows from the PNG, but only %d are left", rows, rows_left);
      goto cleanup;
    }
    
    row_pointers = (png_bytep*)malloc(rows*sizeof(png_bytep));
    for (row=0; row<rows; row++) {
      row_pointers[row] = (png_bytep)PyArray_DATA(arr) + row*PyArray_STRIDE(arr, 0);
    }

    png_read_rows(png_ptr, row_pointers, NULL, rows);
    rows_left -= rows;

    free(row_pointers);
    Py_DECREF(arr);
  }
  
  png_read_end(png_ptr, NULL);

  Py_INCREF(Py_None);
  result = Py_None;

 cleanup:
  if (info_ptr) png_destroy_read_struct (&png_ptr, &info_ptr, NULL);
  if (fp) fclose(fp);
  return result;
}
