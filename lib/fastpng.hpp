#define PNG_SKIP_SETJMP_CHECK
#include "png.h"

#include "lcms2.h"

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

typedef int (*GetScanlinesFunction) (int width, png_bytep *rows_out, int *rowstride_out, void *user_data);

typedef struct {
    PyObject *iterator;
    PyObject *arr_obj;
} PythonScanlineGenerator;

bool
python_scanline_init(PythonScanlineGenerator *self, PyObject *data_generator) {
    self->iterator = PyObject_GetIter(data_generator);
    self->arr_obj = NULL;
    return self->iterator != NULL;
}

void
python_scanline_finalize(PythonScanlineGenerator *self) {
    if (self->iterator) {
        Py_DECREF(self->iterator);
    }
    if (self->arr_obj) {
        Py_DECREF(self->arr_obj);
        self->arr_obj = NULL;
    }
}

int
python_scanline_next(int width, png_bytep *rows_out, int *rowstride_out, void *user_data) {
    PythonScanlineGenerator *self = (PythonScanlineGenerator *)(user_data);

    // Free old array
    if (self->arr_obj) {
        Py_DECREF(self->arr_obj);
        self->arr_obj = NULL;
    }

    self->arr_obj = PyIter_Next(self->iterator);
    if (PyErr_Occurred()) {
        return -1;
    }
    if (!self->arr_obj) {
        return 0;
    }

    PyArrayObject* arr = (PyArrayObject*)self->arr_obj;
    assert(arr); // iterator should have data
    assert(PyArray_ISALIGNED(arr));
    assert(PyArray_NDIM(arr) == 3);
    assert(PyArray_DIM(arr, 1) == width);
    assert(PyArray_DIM(arr, 2) == 4); // rgbu
    assert(PyArray_TYPE(arr) == NPY_UINT8);
    assert(PyArray_STRIDE(arr, 1) == 4);
    assert(PyArray_STRIDE(arr, 2) == 1);

    if (rows_out) {
        *rows_out = (png_bytep)PyArray_DATA(arr);
    }
    if (rowstride_out) {
        *rowstride_out = PyArray_STRIDE(arr, 0);
    }
    return PyArray_DIM(arr, 0);
}

bool
save_png_fast_progressive_c(char *filename, int w, int h,
                            bool has_alpha,
                            bool save_srgb_chunks,
                            GetScanlinesFunction next_scanline_func,
                            void *func_state)

{
  png_structp png_ptr = NULL;
  png_infop info_ptr = NULL;
  bool success = false;

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

  if (save_srgb_chunks) {
    // Internal data is sRGB by the time it gets here.
    // Explicitly save with the recommended chunks to advertise that fact.
    png_set_sRGB_gAMA_and_cHRM (png_ptr, info_ptr, PNG_sRGB_INTENT_PERCEPTUAL);
  }

  // default (all filters enabled):                 1350ms, 3.4MB
  //png_set_filter(png_ptr, 0, PNG_FILTER_NONE);  // 790ms, 3.8MB
  //png_set_filter(png_ptr, 0, PNG_FILTER_PAETH); // 980ms, 3.5MB
  png_set_filter(png_ptr, 0, PNG_FILTER_SUB);     // 760ms, 3.4MB

  //png_set_compression_level(png_ptr, 0); // 0.49s, 32MB
  //png_set_compression_level(png_ptr, 1); // 0.98s, 9.6MB
  png_set_compression_level(png_ptr, 2);   // 1.08s, 9.4MB
  //png_set_compression_level(png_ptr, 9); // 18.6s, 9.3MB

  png_write_info(png_ptr, info_ptr);

  if (!has_alpha) {
    // input array format format is rgbu
    png_set_filler(png_ptr, 0, PNG_FILLER_AFTER);
  }

  {
    int y = 0;
    while (y < h) {
      png_bytep data = NULL;
      int rowstride = -1;
      const int rows = next_scanline_func(w, &data, &rowstride, func_state);
      assert(rows > 0);
      assert(rowstride > 0);
      assert(data);
      y += rows;
      png_bytep p = (png_bytep)data;
      for (int row=0; row<rows; row++) {
        png_write_row(png_ptr, p);
        p += rowstride;
      }
    }
    assert(y == h);
    const int status = next_scanline_func(w, NULL, NULL, func_state);
    assert(status == 0); // iterator should be finished
  }

  png_write_end (png_ptr, NULL);

  success = true;

 cleanup:
  if (info_ptr) png_destroy_write_struct(&png_ptr, &info_ptr);
  if (fp) fclose(fp);
  return success;
}

PyObject *
save_png_fast_progressive (char *filename,
                           int w, int h,
                           bool has_alpha,
                           PyObject *data_generator,
                           bool save_srgb_chunks)
{
    PyObject * result = NULL;
    PythonScanlineGenerator state;
    if (!python_scanline_init(&state, data_generator)) {
        return result;
    }
    const bool success = save_png_fast_progressive_c(filename, w, h, has_alpha, save_srgb_chunks,
                                                     python_scanline_next, (void *)&state);
    if (success) {
        result = Py_BuildValue("{}");
    }
    python_scanline_finalize(&state);
    return result;
}

#ifndef SWIG
static void
png_read_error_callback (png_structp png_read_ptr,
                         png_const_charp error_msg)
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


static const double PNG_gAMA_scale = 100000;
static const double PNG_cHRM_scale = 100000;

static void
log_lcms2_error (cmsContext context_id, cmsUInt32Number err_code,
                 const char *err_text)
{
    printf("lcms: ERROR: %d %s\n", err_code, err_text);
}


/** load_png_fast_progressive:
 *
 * @filename: filename to load, in the system encoding
 * @get_buffer_callback: a Python callable returning writeable arrays
 * @convert_to_srgb: apply colorspace conversions, to sRGB display pixels
 * returns: a dict of flags describing what was read.
 *
 * Read a PNG progressively as 8bit RGBA. The callback must have the signature
 *
 *   numpy_array = callback(full_image_width, full_image_height)
 *
 * @get_buffer_callback  must return a writeable array of the image width.  If
 * the height is smaller than the image height, the callback will be called
 * again until the full image has been processed. The buffer will be written
 * with 8-bit RGBA data
 *
 */

PyObject *
load_png_fast_progressive (char *filename,
                           PyObject *get_buffer_callback,
                           bool convert_to_srgb)
{
  // Note: we are not using the method that libpng calls "Reading PNG
  // files progressively". That method would involve feeding the data
  // into libpng piece by piece, which is not necessary if we can give
  // libpng a simple FILE pointer.

  png_structp png_ptr = NULL;
  png_infop info_ptr = NULL;
  PyObject * result = NULL;
  FILE *fp = NULL;
  uint32_t width, height;
  uint32_t rows_left;
  png_byte color_type;
  png_byte bit_depth;
  bool have_alpha;

  char *cm_processing = NULL;

  // ICC profile-based colour conversion data.
  png_charp icc_profile_name = NULL;
  int icc_compression_type = 0;
#if PNG_LIBPNG_VER < 10500    // 1.5.0beta36, according to libpng CHANGES
  png_charp icc_profile = NULL;
#else
  png_bytep icc_profile = NULL;
#endif
  png_uint_32 icc_proflen = 0;

  // The sRGB flag has an intent field, which we ignore - 
  // the target gamut is sRGB already.
  int srgb_intent = 0;

  // Generic RGB space conversion params.
  // The assumptions we're making are those of sRGB,
  // but they'll be overridden by gammas or primaries in the file if used.
  bool generic_rgb_have_gAMA = false;
  bool generic_rgb_have_cHRM = false;
  double generic_rgb_file_gamma = 45455 / PNG_gAMA_scale;
  double generic_rgb_white_x = 31270 / PNG_cHRM_scale;
  double generic_rgb_white_y = 32900 / PNG_cHRM_scale;
  double generic_rgb_red_x   = 64000 / PNG_cHRM_scale;
  double generic_rgb_red_y   = 33000 / PNG_cHRM_scale;
  double generic_rgb_green_x = 30000 / PNG_cHRM_scale;
  double generic_rgb_green_y = 60000 / PNG_cHRM_scale;
  double generic_rgb_blue_x  = 15000 / PNG_cHRM_scale;
  double generic_rgb_blue_y  =  6000 / PNG_cHRM_scale;

  cmsHPROFILE input_buffer_profile = NULL;
  cmsHPROFILE nparray_data_profile = cmsCreate_sRGBProfile();
  cmsHTRANSFORM input_buffer_to_nparray = NULL;
  cmsToneCurve *gamma_transfer_func = NULL;
  cmsUInt32Number input_buffer_format = 0;

  cmsSetLogErrorHandler(log_lcms2_error);

  fp = fopen(filename, "rb");
  if (!fp) {
    PyErr_SetFromErrno(PyExc_IOError);
    //PyErr_Format(PyExc_IOError, "Could not open PNG file for writing: %s",
    //             filename);
    goto cleanup;
  }

  png_ptr = png_create_read_struct (PNG_LIBPNG_VER_STRING, (png_voidp)NULL,
                                    png_read_error_callback, NULL);
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

if (convert_to_srgb) {
  // If there's an embedded ICC profile, use it in preference to any other
  // colour management information present.
  if (png_get_iCCP (png_ptr, info_ptr, &icc_profile_name,
                    &icc_compression_type, &icc_profile,
                    &icc_proflen))
  {
    input_buffer_profile = cmsOpenProfileFromMem(icc_profile, icc_proflen);
    if (! input_buffer_profile) {
      PyErr_SetString(PyExc_MemoryError, "cmsOpenProfileFromMem() failed");
      goto cleanup;
    }
    cmsColorSpaceSignature cs_sig = cmsGetColorSpace(input_buffer_profile);
    if (cs_sig != cmsSigRgbData) {
      printf("lcms: ignoring non-RGB color profile. "
             "Signature: 0x%08x, '%c%c%c%c'.\n",
             cs_sig,
             0xff&(cs_sig>>24), 0xff&(cs_sig>>16),
             0xff&(cs_sig>>8), 0xff&cs_sig);
      cmsCloseProfile(input_buffer_profile);
      input_buffer_profile = NULL;
    }
  }

  if (input_buffer_profile) {
    cm_processing = "iCCP (use embedded colour profile)";
  }

  // Shorthand for sRGB.
  else if (png_get_sRGB (png_ptr, info_ptr, &srgb_intent)) {
    input_buffer_profile = cmsCreate_sRGBProfile();
    cm_processing = "sRGB (explicit sRGB chunk)";
  }

  else {
    // We might have generic RGB transformation information in the form of
    // the chromaticities for R, G and B and a generic gamma curve.

    if (png_get_cHRM (png_ptr, info_ptr,
                      &generic_rgb_white_x, &generic_rgb_white_y,
                      &generic_rgb_red_x, &generic_rgb_red_y,
                      &generic_rgb_green_x, &generic_rgb_green_y,
                      &generic_rgb_blue_x, &generic_rgb_blue_y))
    {
      generic_rgb_have_cHRM = true;
    }
    if (png_get_gAMA(png_ptr, info_ptr, &generic_rgb_file_gamma)) {
      generic_rgb_have_gAMA = true;
    }
    if (generic_rgb_have_gAMA || generic_rgb_have_cHRM) {
      cmsCIExyYTRIPLE primaries = {{generic_rgb_red_x, generic_rgb_red_y},
                                   {generic_rgb_green_x, generic_rgb_green_y},
                                   {generic_rgb_blue_x, generic_rgb_blue_y}};
      cmsCIExyY white_point = {generic_rgb_white_x, generic_rgb_white_y};
      gamma_transfer_func = cmsBuildGamma(NULL, 1.0/generic_rgb_file_gamma);
      cmsToneCurve *transfer_funcs[3] = {gamma_transfer_func,
                                         gamma_transfer_func,
                                         gamma_transfer_func };
      input_buffer_profile = cmsCreateRGBProfile(&white_point, &primaries,
                                                transfer_funcs);
      cm_processing = "cHRM and/or gAMA (generic RGB space)";
    }

    // Possible legacy PNG, or rather one which might have been written with an
    // old version of MyPaint. Treat as sRGB, but flag the strangeness because
    // it might be important for PNGs in old OpenRaster files.
    else {
      input_buffer_profile = cmsCreate_sRGBProfile();
      cm_processing = "sRGB (no usable CM chunks found)";
    }
  }
} //convert_to_srgb

  if (png_get_interlace_type (png_ptr, info_ptr) != PNG_INTERLACE_NONE) {
    PyErr_SetString(PyExc_RuntimeError,
                    "Interlaced PNG files are not supported!");
    goto cleanup;
  }

  color_type = png_get_color_type(png_ptr, info_ptr);
  bit_depth = png_get_bit_depth(png_ptr, info_ptr);
  have_alpha = color_type & PNG_COLOR_MASK_ALPHA;

  if (color_type == PNG_COLOR_TYPE_PALETTE) {
    png_set_palette_to_rgb(png_ptr);
  }

  if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) {
    png_set_expand_gray_1_2_4_to_8(png_ptr);
  }

  if (png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS)) {
    png_set_tRNS_to_alpha(png_ptr);
    have_alpha = true;
  }

  if (! convert_to_srgb) {
  // Get libpng to convert 16bpp -> 8bpp (LCMS2 does this normally)
  if (bit_depth == 16) {
    png_set_strip_16(png_ptr);
  }
  }//!convert_to_srgb
  if (bit_depth < 8) {
    png_set_packing(png_ptr);
  }

  if (!have_alpha) {
    png_set_add_alpha(png_ptr, 0xFF, PNG_FILLER_AFTER);
  }

  if (color_type == PNG_COLOR_TYPE_GRAY ||
      color_type == PNG_COLOR_TYPE_GRAY_ALPHA) {
    png_set_gray_to_rgb(png_ptr);
  }

  // TODO: do we need gamma transformation here for the no-lcms case?

  png_read_update_info(png_ptr, info_ptr);

  // Verify what we have done
  bit_depth = png_get_bit_depth(png_ptr, info_ptr);
  if (convert_to_srgb) {
  if (! (bit_depth == 8 || bit_depth == 16)) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to convince libpng to convert "
                                        "to 8 or 16 bits per channel");
    goto cleanup;
  }
  } else { //convert_to_srgb
  if (bit_depth != 8) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to convince libpng to convert "
                                        "to 8 bits per channel");
    goto cleanup;
  }
  } //convert_to_srgb
  if (png_get_color_type(png_ptr, info_ptr) != PNG_COLOR_TYPE_RGB_ALPHA) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to convince libpng to convert "
                                        "to RGBA (wrong color_type)");
    goto cleanup;
  }
  if (png_get_channels(png_ptr, info_ptr) != 4) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to convince libpng to convert "
                                        "to RGBA (wrong number of channels)");
    goto cleanup;
  }

if (convert_to_srgb) {
  // PNGs use network byte order, i.e. big-endian in descending order
  // of bit significance. LittleCMS uses whatever's detected for the compiler.
  // ref: http://www.w3.org/TR/2003/REC-PNG-20031110/#7Integers-and-byte-order
  if (bit_depth == 16) {
#ifdef CMS_USE_BIG_ENDIAN
    input_buffer_format = TYPE_RGBA_16;
#else
    input_buffer_format = TYPE_RGBA_16_SE;
#endif
  }
  else {
    input_buffer_format = TYPE_RGBA_8;
  }

  input_buffer_to_nparray = cmsCreateTransform
        (input_buffer_profile, input_buffer_format,
         nparray_data_profile, TYPE_RGBA_8,
         INTENT_PERCEPTUAL, 0);
} //convert_to_srgb

  width = png_get_image_width(png_ptr, info_ptr);
  height = png_get_image_height(png_ptr, info_ptr);
  rows_left = height;

  while (rows_left) {
    PyObject *obj = NULL;
    uint32_t rows = 0;
    uint32_t row = 0;
    // The input buffer is only used when doing color conversions
    const uint8_t input_buf_bytes_per_pixel = (bit_depth==8) ? 4 : 8;
    const uint32_t input_buf_row_stride = sizeof(png_byte) * width
                                          * input_buf_bytes_per_pixel;
    png_byte *input_buffer = NULL;
    // When not converting between colour spaces, the PNG data is
    // written directly to the output rows instead.
    png_bytep *row_pointers = NULL;

    // Invoke the callback to get a chunk of memory to populate
    // Expect it to return a non-contiguous NumPy array
    // with dimensions (h in [1, width]) x (width) x (4)
    obj = PyObject_CallFunction(get_buffer_callback, "ii", width, height);
    if (!obj) {
      PyErr_Format(PyExc_RuntimeError, "Get-buffer callback failed");
      goto cleanup;
    }
    PyArrayObject* pyarr = (PyArrayObject*)obj;
#ifdef HEAVY_DEBUG
    //assert(PyArray_ISCARRAY(arr));
    assert(PyArray_NDIM(pyarr) == 3);
    assert(PyArray_DIM(pyarr, 1) == width);
    assert(PyArray_DIM(pyarr, 2) == 4);
    assert(PyArray_TYPE(pyarr) == NPY_UINT8);
    assert(PyArray_ISBEHAVED(pyarr));
    assert(PyArray_STRIDE(pyarr, 1) == 4*sizeof(uint8_t));
    assert(PyArray_STRIDE(pyarr, 2) ==   sizeof(uint8_t));
#endif
    rows = PyArray_DIM(pyarr, 0);

    if (rows > rows_left) {
      PyErr_Format(PyExc_RuntimeError,
                   "Attempt to read %d rows from the PNG, "
                   "but only %d are left",
                   rows, rows_left);
      goto cleanup;
    }

    row_pointers = (png_bytep *)malloc(rows * sizeof(png_bytep));
if (convert_to_srgb) {
    // rows are 8bpp *or* 16bpp chunks of a temporary input buffer
    input_buffer = (png_byte *) malloc(rows * input_buf_row_stride);
    for (row=0; row<rows; row++) {
      row_pointers[row] = input_buffer + (row * input_buf_row_stride);
    }
} else { //convert_to_srgb
    // rows are always 8bpp chunks of the output NumPy array
    for (row=0; row<rows; row++) {
      row_pointers[row] = (png_bytep)PyArray_DATA(pyarr)
                                     + (row * PyArray_STRIDE(pyarr, 0));
    }
} //convert_to_srgb

    // Populate the strip of memory with pixels decoded from the PNG stream
    png_read_rows(png_ptr, row_pointers, NULL, rows);
    rows_left -= rows;

if (convert_to_srgb) {
    // Apply CMS transform
    for (row=0; row<rows; row++) {
      uint8_t *pyarr_row = (uint8_t *)PyArray_DATA(pyarr)
                         + row*PyArray_STRIDE(pyarr, 0);
      uint8_t *input_row = row_pointers[row];
      // Really minimal fake colour management. Just remaps to sRGB.
      cmsDoTransform(input_buffer_to_nparray, input_row, pyarr_row, width);
      // lcms2 ignores alpha, so copy that verbatim
      // If it's 8bpc RGBA, use A.
      // If it's 16bpc RrGgBbAa, use A.
      for (uint32_t i=0; i<width; ++i) {
        const uint32_t pyarr_alpha_byte = (i*4) + 3;
        const uint32_t buf_alpha_byte = (i*input_buf_bytes_per_pixel)
                                       + ((bit_depth==8) ? 3 : 6);
        pyarr_row[pyarr_alpha_byte] = input_row[buf_alpha_byte];
      }
    }
    free(input_buffer);
} //convert_to_srgb
    free(row_pointers);
    Py_DECREF(obj);
  }

  png_read_end(png_ptr, NULL);

  result = Py_BuildValue("{s:b,s:i,s:i,s:s}",
                         "width", width,
                         "height", height,
                         "cm_conversions_applied", cm_processing);

 cleanup:
  if (info_ptr) png_destroy_read_struct (&png_ptr, &info_ptr, NULL);
  // libpng's style is to free internally allocated stuff like the icc
  // tables in png_destroy_*(). I think.
  if (fp) fclose(fp);
if (convert_to_srgb) {
  if (input_buffer_profile) cmsCloseProfile(input_buffer_profile);
  if (nparray_data_profile) cmsCloseProfile(nparray_data_profile);
  if (input_buffer_to_nparray) cmsDeleteTransform(input_buffer_to_nparray);
  if (gamma_transfer_func) cmsFreeToneCurve(gamma_transfer_func);
} //convert_to_srgb

  return result;
}
