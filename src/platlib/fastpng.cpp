// Fast loading and saving using scalines
// Copyright (C) 2015-2018  The MyPaint Development Team
// Copyright (C) 2008-2014  Martin Renold
//
// This program is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
//
// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
// more details.
//
// You should have received a copy of the GNU General Public License along with
// this program; if not, write to the Free Software Foundation, Inc., 51
// Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.


#include "fastpng.hpp"
#ifdef _WIN32
#ifndef __MINGW64_VERSION_MAJOR
// include this before third party libs
#include <windows.h>
#endif
#endif
#define PNG_SKIP_SETJMP_CHECK
#include "png.h"

#include "lcms2.h"
#include <math.h>
#include <stdint.h>

#include "common.hpp"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>


static void
png_write_error_callback (png_structp png_save_ptr,
                          png_const_charp error_msg)
{
    // we don't trust libpng to call the error callback only once, so
    // check for already-set error
    if (!PyErr_Occurred()) {
        if (!strcmp(error_msg, "Write Error")) {
            PyErr_SetFromErrno(PyExc_IOError);
        }
        else {
            PyErr_Format(PyExc_RuntimeError, "Error writing PNG: %s", error_msg);
        }
    }
    longjmp (png_jmpbuf(png_save_ptr), 1);
}


struct ProgressivePNGWriter::State
{
    int width;
    int height;
    png_structp png_ptr;
    png_infop info_ptr;
    int y;
    PyObject *file;
    FILE *fp;

    State()
        : width(0), height(0),
          png_ptr(NULL), info_ptr(NULL),
          y(0),
          file(NULL),
          fp(NULL)
    { }

    ~State() {
        cleanup();
    }

    bool check_valid();

    void cleanup() {
        if (png_ptr || info_ptr) {
            png_destroy_write_struct(&png_ptr, &info_ptr);
            assert(png_ptr == NULL);
            assert(info_ptr == NULL);
        }
        if (fp) {
#if PY_MAJOR_VERSION >= 3
            fflush(fp);
#endif
            fp = NULL;   // Python code will close it
        }
        if (file) {
            Py_DECREF(file);
            file = NULL;
        }
    }
};


bool
ProgressivePNGWriter::State::check_valid()
{
    bool valid = true;
    if (! info_ptr) {
        PyErr_SetString(
            PyExc_RuntimeError,
            "writer object's internal state is invalid (no info_ptr)"
        );
        valid = false;
    }
    if (! png_ptr) {
        PyErr_SetString(
            PyExc_RuntimeError,
            "writer object's internal state is invalid (no png_ptr)"
        );
        valid = false;
    }
    if (! file) {
        PyErr_SetString(
            PyExc_RuntimeError,
            "writer object's internal state is invalid (no file)"
        );
        valid = false;
    }
    return valid;
}


ProgressivePNGWriter::ProgressivePNGWriter(PyObject *file,
                                           const int w, const int h,
                                           const bool has_alpha,
                                           const bool save_srgb_chunks)
    : state(new ProgressivePNGWriter::State())
{
    state->width = w;
    state->height = h;
    png_structp png_ptr = NULL;
    png_infop info_ptr = NULL;

    const int bpc = 8;

    state->file = file;
    Py_INCREF(file);

#if PY_MAJOR_VERSION >= 3
    // See https://docs.python.org/3.5/c-api/file.html
    // Also https://stackoverflow.com/a/40598787
    int fd = PyObject_AsFileDescriptor(file);
    if (fd == -1) {
        PyErr_SetString(
            PyExc_TypeError,
            "file arg is not an int, or it has no fileno() method"
        );
        state->cleanup();
        return;
    }
    FILE *fp = fdopen(fd, "w");
#else
    if (! PyFile_Check(file)) {
        PyErr_SetString(
            PyExc_TypeError,
            "file arg must be a builtin file object"
        );
        state->cleanup();
        return;
    }
    FILE *fp = PyFile_AsFile(file);
#endif
    if (!fp) {
        PyErr_SetString(
            PyExc_TypeError,
            "file arg has no file descriptor or FILE* associated with it"
        );
        state->cleanup();
        return;
    }
    state->fp = fp;

    png_ptr = png_create_write_struct (PNG_LIBPNG_VER_STRING,
                                       (png_voidp)NULL,
                                       png_write_error_callback,
                                       NULL);
    if (!png_ptr) {
        PyErr_SetString(PyExc_MemoryError, "png_create_write_struct() failed");
        state->cleanup();
        return;
    }
    state->png_ptr = png_ptr;

    info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        PyErr_SetString(PyExc_MemoryError, "png_create_info_struct() failed");
        state->cleanup();
        return;
    }
    state->info_ptr = info_ptr;

    if (! state->check_valid()) {
        state->cleanup();
        return;
    }

    if (setjmp(png_jmpbuf(png_ptr))) {
        PyErr_SetString(PyExc_RuntimeError, "libpng error during constructor");
        state->cleanup();
        return;
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
        png_set_sRGB_gAMA_and_cHRM (png_ptr, info_ptr,
                                    PNG_sRGB_INTENT_PERCEPTUAL);
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
}


PyObject *
ProgressivePNGWriter::write(PyObject *arr_obj)
{
    PyArrayObject* arr = (PyArrayObject*)arr_obj;
    int rowcount = 0;
    int rowstride = 0;
    png_bytep rowdata = NULL;
    png_bytep row_p = NULL;
    int row = 0;
    char *err_text = NULL;
    PyObject *err_type = PyExc_RuntimeError;

    if (! state) {
        err_type = PyExc_RuntimeError;
        err_text = "writer object is not ready to write (internal state lost)";
        goto errexit;
    }
    if (! state->check_valid()) {
        state->cleanup();
        return NULL;
    }

    if (!arr_obj || !PyArray_Check(arr_obj)) {
        err_type = PyExc_TypeError;
        err_text = "arg must be a numpy array (of HxWx4)";
        goto errexit;
    }
    if (!PyArray_ISALIGNED(arr) || PyArray_NDIM(arr)!=3) {
        err_type = PyExc_ValueError;
        err_text = "arg must be an aligned HxWx4 numpy array";
        goto errexit;
    }
    if (PyArray_DIM(arr, 1) != state->width) {
        err_type = PyExc_ValueError;
        err_text = "strip width must match writer width (must be HxWx4)";
        goto errexit;
    }
    if (PyArray_DIM(arr, 2) != 4) {
        err_type = PyExc_ValueError;
        err_text = "strip must contain RGBA data (must be HxWx4)";
        goto errexit;
    }
    if (PyArray_TYPE(arr) != NPY_UINT8) {
        err_type = PyExc_ValueError;
        err_text = "strip must contain uint8 RGBA only";
        goto errexit;
    }
    assert(PyArray_STRIDE(arr, 1) == 4);
    assert(PyArray_STRIDE(arr, 2) == 1);

    if (setjmp(png_jmpbuf(state->png_ptr))) {
        if (PyErr_Occurred()) {
            state->cleanup();
            return NULL;
        }
        err_type = PyExc_RuntimeError;
        err_text = "libpng error during write()";
        goto errexit;
    }
    rowcount = PyArray_DIM(arr, 0);
    rowstride = PyArray_STRIDE(arr, 0);
    rowdata = (png_bytep)PyArray_DATA(arr);
    row_p = (png_bytep)rowdata;
    for (row=0; row<rowcount; row++) {
        png_write_row(state->png_ptr, row_p);
        if (! state->check_valid()) {
            state->cleanup();
            return NULL;
        }
        row_p += rowstride;
        if (++(state->y) > state->height) {
            err_type = PyExc_RuntimeError;
            err_text = "too many pixel rows written";
            goto errexit;
        }
    }
    Py_RETURN_NONE;

  errexit:
    if (state) {
        state->cleanup();
    }
    if (err_text) {
        PyErr_SetString(err_type, err_text);
        return NULL;
    }
    Py_RETURN_NONE;
}


PyObject *
ProgressivePNGWriter::close()
{
    if (! state) {
        PyErr_SetString(
            PyExc_RuntimeError,
            "writer object is not ready to write (internal state lost)"
        );
        return NULL;
    }
    if (! state->check_valid()) {
        state->cleanup();
        return NULL;
    }
    if (setjmp(png_jmpbuf(state->png_ptr))) {
        state->cleanup();
        PyErr_SetString(PyExc_RuntimeError, "libpng error during close()");
        return NULL;
    }
    png_write_end (state->png_ptr, NULL);
    if (state->y != state->height) {
        state->cleanup();
        PyErr_SetString(
            PyExc_RuntimeError,
            "too many pixel rows written"
        );
        return NULL;
    }
    state->cleanup();
    Py_RETURN_NONE;
}


ProgressivePNGWriter::~ProgressivePNGWriter()
{
    delete state;
}


static void
png_read_error_callback (png_structp png_read_ptr,
                         png_const_charp error_msg)
{
    // we don't trust libpng to call the error callback only once, so
    // check for already-set error
    if (!PyErr_Occurred()) {
        if (!strcmp(error_msg, "Read Error")) {
            PyErr_SetFromErrno(PyExc_IOError);
        }
        else {
            PyErr_Format(PyExc_RuntimeError,
                         "Error reading PNG: %s",
                         error_msg);
        }
    }
    longjmp (png_jmpbuf(png_read_ptr), 1);
}


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
    PyObject *result = NULL;
    FILE *fp = NULL;
    uint32_t width, height;
    uint32_t rows_left;
    png_byte color_type;
    png_byte bit_depth;
    bool have_alpha;

    // Textual description of what processing was applied and why
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
    double generic_rgb_file_gamma = 1.0 / 2.2 ;   //45455 / PNG_gAMA_scale;
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

#ifdef _WIN32
    wchar_t *win32_filename;
#ifdef __MINGW64_VERSION_MAJOR
    // mbstowcs seems mismatch with default python encoding, force to be utf8
    __mingw_str_utf8_wide(filename, &win32_filename, NULL);
#else
    size_t len;
    wchar_t *buf;
    // what __mingw_str_utf8_wide is
    len = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, filename, -1, NULL, 0); 
    buf = (wchar_t *) calloc(len + 1, sizeof (wchar_t));
    if(!buf)
        len = 0;
    else {
        if (len != 0)
            MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, filename, -1, buf, len);
        buf[len] = L'0'; // Must null-terminated
    }
    win32_filename = buf;
#endif
    fp = _wfopen(win32_filename, L"rb");
    if (win32_filename)
        free(win32_filename);
#else
    fp = fopen(filename, "rb");
#endif
    if (!fp) {
        PyErr_SetFromErrno(PyExc_IOError);
        goto cleanup;
    }

    png_ptr = png_create_read_struct (PNG_LIBPNG_VER_STRING, (png_voidp)NULL,
                                      png_read_error_callback, NULL);
    if (!png_ptr) {
        PyErr_SetString(PyExc_MemoryError, "png_create_read_struct() failed");
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
        // If there's an embedded ICC profile, but only if it's an RGB one,
        // use it in preference to any other colour management chunks present.
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
            cm_processing = "Converted from a calibrated colorspace using an embedded ICC profile";
        }

        // 
        else if (png_get_sRGB (png_ptr, info_ptr, &srgb_intent)) {
            cm_processing = "None: image was tagged as sRGB";
            convert_to_srgb = false;
        }

        else {
            // We might have generic RGB transformation information in
            // the form of the chromaticities for R, G, B, and a
            // generic gamma curve.

            if (png_get_cHRM(png_ptr, info_ptr,
                             &generic_rgb_white_x, &generic_rgb_white_y,
                             &generic_rgb_red_x, &generic_rgb_red_y,
                             &generic_rgb_green_x, &generic_rgb_green_y,
                             &generic_rgb_blue_x, &generic_rgb_blue_y)) {
                generic_rgb_have_cHRM = true;
            }
            if (png_get_gAMA(png_ptr, info_ptr, &generic_rgb_file_gamma)) {
                generic_rgb_have_gAMA = true;
            }

            // The PNG spec specifically says:
            //
            // "RGB samples represent calibrated colour information
            // if the colour space is indicated (by gAMA and cHRM,
            // or sRGB, or iCCP) or uncalibrated device-dependent
            // colour if not" -- http://www.w3.org/TR/PNG/#6Colour-values
            //
            // But we can't really interpret that as "if and only
            // if" because if we do then pngsuite test images will
            // fail.

            if (generic_rgb_have_gAMA
                && ! generic_rgb_have_cHRM
                && fabs(generic_rgb_file_gamma - 1.0/2.2) < 0.01)
            {
                // This is quite likely to be saved by GIMP,
                // whose default "sRGB built-in" colorspace
                // is saved with a gAMA of 1/2.2,
                // with no sRGB or cHRM chunks.
                cm_processing = "None: assumed to be GIMP's "
                    "default \"sRGB built-in\" colorspace: "
                    "gAMA~=1/2.2 and no other colorimetric chunks";
                convert_to_srgb = false;
            }
            else if (generic_rgb_have_gAMA || generic_rgb_have_cHRM) {

                cmsCIExyYTRIPLE primaries = {
                    {generic_rgb_red_x, generic_rgb_red_y},
                    {generic_rgb_green_x, generic_rgb_green_y},
                    {generic_rgb_blue_x, generic_rgb_blue_y}
                };
                cmsCIExyY white_point = {
                    generic_rgb_white_x,
                    generic_rgb_white_y
                };
                gamma_transfer_func = cmsBuildGamma(
                    NULL,
                    1.0/generic_rgb_file_gamma
                );
                cmsToneCurve *transfer_funcs[3] = {
                    gamma_transfer_func,
                    gamma_transfer_func,
                    gamma_transfer_func
                };
                input_buffer_profile = cmsCreateRGBProfile(
                    &white_point,
                    &primaries,
                    transfer_funcs
                );

                if (!generic_rgb_have_cHRM) {
                    cm_processing = "Converted from a generic RGB space "
                        "described by the file's gAMA chunk, "
                        "but assuming sRGB primaries because "
                        "no cHRM chunk was present";
                }
                else if (! generic_rgb_have_gAMA) {
                    cm_processing = "Converted from a generic RGB space "
                        "described by the file's gAMA chunk, "
                        "but assuming an sRGB-like tone curve "
                        "in the absence of a gAMA chunk";
                }
                else {
                    cm_processing = "Converted from a generic RGB space "
                       "described by the file's cHRM and gAMA chunks";
                }
            }
            else {
                input_buffer_profile = cmsCreate_sRGBProfile();
                cm_processing = "None: no usable colorimetric chunks were found";
                convert_to_srgb = false;
            }
        }
    } //convert_to_srgb

    if (png_get_interlace_type (png_ptr, info_ptr) != PNG_INTERLACE_NONE) {
        PyErr_SetString(
            PyExc_RuntimeError,
            "Interlaced PNG files are not supported!"
        );
        goto cleanup;
    }

    // Set PNG loader flags

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
    }

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

    png_read_update_info(png_ptr, info_ptr);

    // Verify what we have done
    bit_depth = png_get_bit_depth(png_ptr, info_ptr);
    if (convert_to_srgb) {
        if (! (bit_depth == 8 || bit_depth == 16)) {
            PyErr_SetString(
                PyExc_RuntimeError,
                "Failed to convince libpng to convert "
                "to 8 or 16 bits per channel"
            );
            goto cleanup;
        }
    }
    else {
        if (bit_depth != 8) {
            PyErr_SetString(
                PyExc_RuntimeError,
                "Failed to convince libpng to convert "
                "to 8 bits per channel"
            );
            goto cleanup;
        }
    } //convert_to_srgb
    if (png_get_color_type(png_ptr, info_ptr) != PNG_COLOR_TYPE_RGB_ALPHA) {
        PyErr_SetString(
            PyExc_RuntimeError,
            "Failed to convince libpng to convert "
            "to RGBA (wrong color_type)"
        );
        goto cleanup;
    }
    if (png_get_channels(png_ptr, info_ptr) != 4) {
        PyErr_SetString(
            PyExc_RuntimeError,
            "Failed to convince libpng to convert "
            "to RGBA (wrong number of channels)"
        );
        goto cleanup;
    }

    if (convert_to_srgb && input_buffer_profile) {
        // PNGs use network byte order, i.e. big-endian in descending
        // order of bit significance. LittleCMS uses whatever's detected
        // for the compiler.
        // http://www.w3.org/TR/2003/REC-PNG-20031110/#7Integers-and-byte-order
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
        input_buffer_to_nparray = cmsCreateTransform(
            input_buffer_profile, input_buffer_format,
            nparray_data_profile, TYPE_RGBA_8,
            INTENT_PERCEPTUAL,
            0
        );
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
        }
        else {
            // rows are always 8bpp chunks of the output NumPy array
            for (row=0; row<rows; row++) {
                row_pointers[row] = (png_bytep)PyArray_DATA(pyarr)
                                  + (row * PyArray_STRIDE(pyarr, 0));
            }
        }

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
                cmsDoTransform(
                    input_buffer_to_nparray,
                    input_row,
                    pyarr_row,
                    width
                );
                // lcms2 ignores alpha, so copy that verbatim
                // If it's 8bpc RGBA, use A.
                // If it's 16bpc RrGgBbAa, use A.
                for (uint32_t i=0; i<width; ++i) {
                    const uint32_t pyarr_alpha_byte = (i*4) + 3;
                    const uint32_t buf_alpha_byte =
                        (i*input_buf_bytes_per_pixel)
                        + ((bit_depth==8) ? 3 : 6);
                    pyarr_row[pyarr_alpha_byte] = input_row[buf_alpha_byte];
                }
            }
            free(input_buffer);
        }
        free(row_pointers);
        Py_DECREF(obj);
    } //while (rows_left)

    png_read_end(png_ptr, NULL);

    result = Py_BuildValue(
        "{s:i,s:i,s:s,s:b}",
        "width", width,
        "height", height,
        "cm_transform_desc", cm_processing,
        "cm_transformed_to_srgb", convert_to_srgb
    );

cleanup:
    if (info_ptr || info_ptr) {
        png_destroy_read_struct (&png_ptr, &info_ptr, NULL);
    }
    // libpng's style is to free internally allocated stuff like the icc
    // tables in png_destroy_*(). I think.
    if (fp)
        fclose(fp);
    if (convert_to_srgb) {
        if (input_buffer_profile)
            cmsCloseProfile(input_buffer_profile);
        if (nparray_data_profile)
            cmsCloseProfile(nparray_data_profile);
        if (input_buffer_to_nparray)
            cmsDeleteTransform(input_buffer_to_nparray);
        if (gamma_transfer_func)
            cmsFreeToneCurve(gamma_transfer_func);
    }

    return result;
}
