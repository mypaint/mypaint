/* This file is part of MyPaint.
 * Copyright (C) 2008-2014 by Martin Renold <martinxyz@gmx.ch>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */

#include "pixops.hpp"

#include "common.hpp"
#include "compositing.hpp"
#include "blending.hpp"
#include "fastapprox/fastpow.h"

#include <mypaint-tiled-surface.h>

#include <glib.h>
#include <math.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

#include <stdlib.h>
#include <math.h>


void
tile_downscale_rgba16_c(const uint16_t *src, int src_strides, uint16_t *dst,
                        int dst_strides, int dst_x, int dst_y)
{
  for (int y=0; y<MYPAINT_TILE_SIZE/2; y++) {
    uint16_t * src_p = (uint16_t*)((char *)src + (2*y)*src_strides);
    uint16_t * dst_p = (uint16_t*)((char *)dst + (y+dst_y)*dst_strides);
    dst_p += 4*dst_x;
    for(int x=0; x<MYPAINT_TILE_SIZE/2; x++) {
      dst_p[0] = src_p[0]/4 + (src_p+4)[0]/4 + (src_p+4*MYPAINT_TILE_SIZE)[0]/4 + (src_p+4*MYPAINT_TILE_SIZE+4)[0]/4;
      dst_p[1] = src_p[1]/4 + (src_p+4)[1]/4 + (src_p+4*MYPAINT_TILE_SIZE)[1]/4 + (src_p+4*MYPAINT_TILE_SIZE+4)[1]/4;
      dst_p[2] = src_p[2]/4 + (src_p+4)[2]/4 + (src_p+4*MYPAINT_TILE_SIZE)[2]/4 + (src_p+4*MYPAINT_TILE_SIZE+4)[2]/4;
      dst_p[3] = src_p[3]/4 + (src_p+4)[3]/4 + (src_p+4*MYPAINT_TILE_SIZE)[3]/4 + (src_p+4*MYPAINT_TILE_SIZE+4)[3]/4;
      src_p += 8;
      dst_p += 4;
    }
  }
}

void tile_downscale_rgba16(PyObject *src, PyObject *dst, int dst_x, int dst_y) {

  PyArrayObject* src_arr = ((PyArrayObject*)src);
  PyArrayObject* dst_arr = ((PyArrayObject*)dst);

#ifdef HEAVY_DEBUG
  assert(PyArray_Check(src));
  assert(PyArray_DIM(src_arr, 0) == MYPAINT_TILE_SIZE);
  assert(PyArray_DIM(src_arr, 1) == MYPAINT_TILE_SIZE);
  assert(PyArray_DIM(src_arr, 2) == 4);
  assert(PyArray_TYPE(src_arr) == NPY_UINT16);
  assert(PyArray_ISCARRAY(src_arr));

  assert(PyArray_Check(dst));
  assert(PyArray_DIM(dst_arr, 2) == 4);
  assert(PyArray_TYPE(dst_arr) == NPY_UINT16);
  assert(PyArray_ISCARRAY(dst_arr));
#endif

  tile_downscale_rgba16_c((uint16_t*)PyArray_DATA(src_arr), PyArray_STRIDES(src_arr)[0],
                          (uint16_t*)PyArray_DATA(dst_arr), PyArray_STRIDES(dst_arr)[0],
                          dst_x, dst_y);

}


void tile_copy_rgba16_into_rgba16_c(const uint16_t *src, uint16_t *dst) {
  memcpy(dst, src, MYPAINT_TILE_SIZE*MYPAINT_TILE_SIZE*4*sizeof(uint16_t));
}

void tile_copy_rgba16_into_rgba16(PyObject * src, PyObject * dst) {
  PyArrayObject* src_arr = ((PyArrayObject*)src);
  PyArrayObject* dst_arr = ((PyArrayObject*)dst);

#ifdef HEAVY_DEBUG
  assert(PyArray_Check(dst));
  assert(PyArray_DIM(dst_arr, 0) == MYPAINT_TILE_SIZE);
  assert(PyArray_DIM(dst_arr, 1) == MYPAINT_TILE_SIZE);
  assert(PyArray_DIM(dst_arr, 2) == 4);
  assert(PyArray_TYPE(dst_arr) == NPY_UINT16);
  assert(PyArray_ISCARRAY(dst_arr));
  assert(PyArray_STRIDES(dst_arr)[1] == 4*sizeof(uint16_t));
  assert(PyArray_STRIDES(dst_arr)[2] ==   sizeof(uint16_t));

  assert(PyArray_Check(src));
  assert(PyArray_DIM(src_arr, 0) == MYPAINT_TILE_SIZE);
  assert(PyArray_DIM(src_arr, 1) == MYPAINT_TILE_SIZE);
  assert(PyArray_DIM(src_arr, 2) == 4);
  assert(PyArray_TYPE(src_arr) == NPY_UINT16);
  assert(PyArray_ISCARRAY(src_arr));
  assert(PyArray_STRIDES(src_arr)[1] == 4*sizeof(uint16_t));
  assert(PyArray_STRIDES(src_arr)[2] ==   sizeof(uint16_t));
#endif

  /* the code below can be used if it is not ISCARRAY, but only ISBEHAVED:
  char * src_p = PyArray_DATA(src_arr);
  char * dst_p = PyArray_DATA(dst_arr);
  for (int y=0; y<MYPAINT_TILE_SIZE; y++) {
    memcpy(dst_p, src_p, MYPAINT_TILE_SIZE*4);
    src_p += src_arr->strides[0];
    dst_p += dst_arr->strides[0];
  }
  */

  tile_copy_rgba16_into_rgba16_c((uint16_t *)PyArray_DATA(src_arr),
                                 (uint16_t *)PyArray_DATA(dst_arr));
}

void tile_clear_rgba8(PyObject * dst) {
  PyArrayObject* dst_arr = ((PyArrayObject*)dst);

#ifdef HEAVY_DEBUG
  assert(PyArray_Check(dst));
  assert(PyArray_DIM(dst_arr, 0) == MYPAINT_TILE_SIZE);
  assert(PyArray_DIM(dst_arr, 1) == MYPAINT_TILE_SIZE);
  assert(PyArray_TYPE(dst_arr) == NPY_UINT8);
  assert(PyArray_ISBEHAVED(dst_arr));
  assert(PyArray_STRIDES(dst_arr)[1] <= 8);
#endif

  for (int y=0; y<MYPAINT_TILE_SIZE; y++) {
    uint8_t  * dst_p = (uint8_t*)((char *)PyArray_DATA(dst_arr) + y*PyArray_STRIDES(dst_arr)[0]);
    memset(dst_p, 0, MYPAINT_TILE_SIZE*PyArray_STRIDES(dst_arr)[1]);
    dst_p += PyArray_STRIDES(dst_arr)[0];
  }
}

void tile_clear_rgba16(PyObject * dst) {
  PyArrayObject* dst_arr = ((PyArrayObject*)dst);

#ifdef HEAVY_DEBUG
  assert(PyArray_Check(dst));
  assert(PyArray_DIM(dst_arr, 0) == MYPAINT_TILE_SIZE);
  assert(PyArray_DIM(dst_arr, 1) == MYPAINT_TILE_SIZE);
  assert(PyArray_TYPE(dst_arr) == NPY_UINT16);
  assert(PyArray_ISBEHAVED(dst_arr));
  assert(PyArray_STRIDES(dst_arr)[1] <= 8);
#endif

  for (int y=0; y<MYPAINT_TILE_SIZE; y++) {
    uint16_t  * dst_p = (uint16_t*)((char *)PyArray_DATA(dst_arr) + y*PyArray_STRIDES(dst_arr)[0]);
    memset(dst_p, 0, MYPAINT_TILE_SIZE*PyArray_STRIDES(dst_arr)[1]);
    dst_p += PyArray_STRIDES(dst_arr)[0];
  }
}


// Noise used for dithering (the same for each tile).

static const int dithering_noise_size = MYPAINT_TILE_SIZE*MYPAINT_TILE_SIZE*4;
static uint16_t dithering_noise[dithering_noise_size];
static void precalculate_dithering_noise_if_required()
{
  static bool have_noise = false;
  if (!have_noise) {
    // let's make some noise
    for (int i=0; i<dithering_noise_size; i++) {
      // random number in range [0.03 .. 0.2] * (1<<15)
      //
      // We could use the full range, but like this it is much easier
      // to guarantee 8bpc load-save roundtrips don't alter the
      // image. With the full range we would have to pay a lot
      // attention to rounding converting 8bpc to our internal format.
      dithering_noise[i] = (rand() % (1<<15)) * 5/256 + (1<<15) * 2/256;
    }
    have_noise = true;
  }
}

// Used for saving layers (transparent PNG), and for display when there
// can be transparent areas in the output.

static inline void
tile_convert_rgba16_to_rgba8_c (const uint16_t* const src,
                                const int src_strides,
                                const uint8_t* dst,
                                const int dst_strides,
                                const float EOTF)
{
  precalculate_dithering_noise_if_required();

  for (int y=0; y<MYPAINT_TILE_SIZE; y++) {
    int noise_idx = y*MYPAINT_TILE_SIZE*4;
    const uint16_t *src_p = (uint16_t*)((char *)src + y*src_strides);
    uint8_t *dst_p = (uint8_t*)((char *)dst + y*dst_strides);
    for (int x=0; x<MYPAINT_TILE_SIZE; x++) {
      uint32_t r, g, b, a;
      r = *src_p++;
      g = *src_p++;
      b = *src_p++;
      a = *src_p++;
#ifdef HEAVY_DEBUG
      assert(a<=(1<<15));
      assert(r<=(1<<15));
      assert(g<=(1<<15));
      assert(b<=(1<<15));
      assert(r<=a);
      assert(g<=a);
      assert(b<=a);
#endif
      // un-premultiply alpha (with rounding)
      if (a != 0) {
        const uint32_t rnd_a = a/2;
        r = ((r << 15) + rnd_a) / a;
        g = ((g << 15) + rnd_a) / a;
        b = ((b << 15) + rnd_a) / a;
      } else {
        r = g = b = 0;
      }
#ifdef HEAVY_DEBUG
      assert(a<=(1<<15));
      assert(r<=(1<<15));
      assert(g<=(1<<15));
      assert(b<=(1<<15));
#endif
      /*
      // Variant A) rounding
      const uint32_t add_r = (1<<15)/2;
      const uint32_t add_g = (1<<15)/2;
      const uint32_t add_b = (1<<15)/2;
      const uint32_t add_a = (1<<15)/2;
      */

      /*
      // Variant B) naive dithering
      // This can alter the alpha channel during a load->save cycle.
      const uint32_t add_r = rand() % (1<<15);
      const uint32_t add_g = rand() % (1<<15);
      const uint32_t add_b = rand() % (1<<15);
      const uint32_t add_a = rand() % (1<<15);
      */

      /*
      // Variant C) slightly better dithering
      // make sure we don't dither rounding errors (those did occur when converting 8bit-->16bit)
      // this preserves the alpha channel, but we still add noise to the highly transparent colors
      const uint32_t add_r = (rand() % (1<<15)) * 240/256 + (1<<15) * 8/256;
      const uint32_t add_g = add_r; // hm... do not produce too much color noise
      const uint32_t add_b = add_r;
      const uint32_t add_a = (rand() % (1<<15)) * 240/256 + (1<<15) * 8/256;
      // TODO: error diffusion might work better than random dithering...
      */

      // Variant C) but with precalculated noise (much faster)
      //
      const float add_r = (float)dithering_noise[noise_idx+0] / (1<<30);
      //const uint32_t add_g = add_r; // hm... do not produce too much color noise
      //const uint32_t add_b = add_r;
      const uint32_t add_a = dithering_noise[noise_idx+1];
      noise_idx += 4;

#ifdef HEAVY_DEBUG
      assert(add_a < (1<<15));
      assert(add_a >= 0);
      assert(noise_idx <= dithering_noise_size);
#endif

      *dst_p++ = uint8_t(fastpow((float)r / (1<<15) + add_r, 1.0/EOTF) * 255);
      *dst_p++ = uint8_t(fastpow((float)g / (1<<15) + add_r, 1.0/EOTF) * 255);
      *dst_p++ = uint8_t(fastpow((float)b / (1<<15) + add_r, 1.0/EOTF) * 255);
      *dst_p++ = ((a * 255 + add_a) / (1<<15));
    }
    src_p += src_strides;
    dst_p += dst_strides;
  }
}



void
tile_convert_rgba16_to_rgba8 (PyObject *src,
                              PyObject *dst, const float EOTF)
{
  PyArrayObject* src_arr = ((PyArrayObject*)src);
  PyArrayObject* dst_arr = ((PyArrayObject*)dst);

#ifdef HEAVY_DEBUG
  assert(PyArray_Check(dst));
  assert(PyArray_DIM(dst_arr, 0) == MYPAINT_TILE_SIZE);
  assert(PyArray_DIM(dst_arr, 1) == MYPAINT_TILE_SIZE);
  assert(PyArray_DIM(dst_arr, 2) == 4);
  assert(PyArray_TYPE(dst_arr) == NPY_UINT8);
  assert(PyArray_ISBEHAVED(dst_arr));
  assert(PyArray_STRIDE(dst_arr, 1) == 4*sizeof(uint8_t));
  assert(PyArray_STRIDE(dst_arr, 2) == sizeof(uint8_t));

  assert(PyArray_Check(src));
  assert(PyArray_DIM(src_arr, 0) == MYPAINT_TILE_SIZE);
  assert(PyArray_DIM(src_arr, 1) == MYPAINT_TILE_SIZE);
  assert(PyArray_DIM(src_arr, 2) == 4);
  assert(PyArray_TYPE(src_arr) == NPY_UINT16);
  assert(PyArray_ISBEHAVED(src_arr));
  assert(PyArray_STRIDE(src_arr, 1) == 4*sizeof(uint16_t));
  assert(PyArray_STRIDE(src_arr, 2) ==   sizeof(uint16_t));
#endif

  tile_convert_rgba16_to_rgba8_c((uint16_t*)PyArray_DATA(src_arr),
                                 PyArray_STRIDES(src_arr)[0],
                                 (uint8_t*)PyArray_DATA(dst_arr),
                                 PyArray_STRIDES(dst_arr)[0],
                                 EOTF);
}

static inline void
tile_convert_rgbu16_to_rgbu8_c(const uint16_t* const src,
                               const int src_strides,
                               const uint8_t* dst,
                               const int dst_strides,
                               const float EOTF)
{
  precalculate_dithering_noise_if_required();

  for (int y=0; y<MYPAINT_TILE_SIZE; y++) {
    int noise_idx = y*MYPAINT_TILE_SIZE*4;
    const uint16_t *src_p = (uint16_t*)((char *)src + y*src_strides);
    uint8_t *dst_p = (uint8_t*)((char *)dst + y*dst_strides);
    for (int x=0; x<MYPAINT_TILE_SIZE; x++) {
      float r, g, b;
      r = ((float)*src_p++ / (1<<15));
      g = ((float)*src_p++ / (1<<15));
      b = ((float)*src_p++ / (1<<15));
      src_p++; // alpha unused
#ifdef HEAVY_DEBUG
      assert(r<=(1<<15));
      assert(g<=(1<<15));
      assert(b<=(1<<15));
#endif

      /*
      // rounding
      const uint32_t add = (1<<15)/2;
      */
      // dithering
      const float add = (float)dithering_noise[noise_idx++] / (1<<30);

      *dst_p++ = (fastpow(r + add, 1.0/EOTF) ) * 255 + 0.5;
      *dst_p++ = (fastpow(g + add, 1.0/EOTF) ) * 255 + 0.5;
      *dst_p++ = (fastpow(b + add, 1.0/EOTF) ) * 255 + 0.5;
      *dst_p++ = 255;
    }
#ifdef HEAVY_DEBUG
    assert(noise_idx <= dithering_noise_size);
#endif
    src_p += src_strides;
    dst_p += dst_strides;
  }
}


void tile_convert_rgbu16_to_rgbu8(PyObject * src, PyObject * dst, const float EOTF) {
  PyArrayObject* src_arr = ((PyArrayObject*)src);
  PyArrayObject* dst_arr = ((PyArrayObject*)dst);

#ifdef HEAVY_DEBUG
  assert(PyArray_Check(dst));
  assert(PyArray_DIM(dst_arr, 0) == MYPAINT_TILE_SIZE);
  assert(PyArray_DIM(dst_arr, 1) == MYPAINT_TILE_SIZE);
  assert(PyArray_DIM(dst_arr, 2) == 4);
  assert(PyArray_TYPE(dst_arr) == NPY_UINT8);
  assert(PyArray_ISBEHAVED(dst_arr));
  assert(PyArray_STRIDE(dst_arr, 1) == 4*sizeof(uint8_t));
  assert(PyArray_STRIDE(dst_arr, 2) == sizeof(uint8_t));

  assert(PyArray_Check(src));
  assert(PyArray_DIM(src_arr, 0) == MYPAINT_TILE_SIZE);
  assert(PyArray_DIM(src_arr, 1) == MYPAINT_TILE_SIZE);
  assert(PyArray_DIM(src_arr, 2) == 4);
  assert(PyArray_TYPE(src_arr) == NPY_UINT16);
  assert(PyArray_ISBEHAVED(src_arr));
  assert(PyArray_STRIDE(src_arr, 1) == 4*sizeof(uint16_t));
  assert(PyArray_STRIDE(src_arr, 2) ==   sizeof(uint16_t));
#endif

  tile_convert_rgbu16_to_rgbu8_c((uint16_t*)PyArray_DATA(src_arr), PyArray_STRIDES(src_arr)[0],
                                 (uint8_t*)PyArray_DATA(dst_arr), PyArray_STRIDES(dst_arr)[0],
                                  EOTF);
}


// used mainly for loading layers (transparent PNG)
void tile_convert_rgba8_to_rgba16(PyObject * src, PyObject * dst, const float EOTF) {
  PyArrayObject* src_arr = ((PyArrayObject*)src);
  PyArrayObject* dst_arr = ((PyArrayObject*)dst);

#ifdef HEAVY_DEBUG
  assert(PyArray_Check(dst));
  assert(PyArray_DIM(dst_arr, 0) == MYPAINT_TILE_SIZE);
  assert(PyArray_DIM(dst_arr, 1) == MYPAINT_TILE_SIZE);
  assert(PyArray_DIM(dst_arr, 2) == 4);
  assert(PyArray_TYPE(dst_arr) == NPY_UINT16);
  assert(PyArray_ISBEHAVED(dst_arr));
  assert(PyArray_STRIDES(dst_arr)[1] == 4*sizeof(uint16_t));
  assert(PyArray_STRIDES(dst_arr)[2] ==   sizeof(uint16_t));

  assert(PyArray_Check(src));
  assert(PyArray_DIM(src_arr, 0) == MYPAINT_TILE_SIZE);
  assert(PyArray_DIM(src_arr, 1) == MYPAINT_TILE_SIZE);
  assert(PyArray_DIM(src_arr, 2) == 4);
  assert(PyArray_TYPE(src_arr) == NPY_UINT8);
  assert(PyArray_ISBEHAVED(src_arr));
  assert(PyArray_STRIDES(src_arr)[1] == 4*sizeof(uint8_t));
  assert(PyArray_STRIDES(src_arr)[2] ==   sizeof(uint8_t));
#endif

  for (int y=0; y<MYPAINT_TILE_SIZE; y++) {
    uint8_t  * src_p = (uint8_t*)((char *)PyArray_DATA(src_arr) + y*PyArray_STRIDES(src_arr)[0]);
    uint16_t * dst_p = (uint16_t*)((char *)PyArray_DATA(dst_arr) + y*PyArray_STRIDES(dst_arr)[0]);
    for (int x=0; x<MYPAINT_TILE_SIZE; x++) {
      uint32_t r, g, b, a;
      r = *src_p++;
      g = *src_p++;
      b = *src_p++;
      a = *src_p++;

      // convert to fixed point (with rounding)
      r = uint32_t(fastpow((float)r/255.0, EOTF) * (1<<15) + 0.5);
      g = uint32_t(fastpow((float)g/255.0, EOTF) * (1<<15) + 0.5);
      b = uint32_t(fastpow((float)b/255.0, EOTF) * (1<<15) + 0.5);
      a = (a * (1<<15) + 255/2) / 255;

      // premultiply alpha (with rounding), save back
      *dst_p++ = (r * a + (1<<15)/2) / (1<<15);
      *dst_p++ = (g * a + (1<<15)/2) / (1<<15);
      *dst_p++ = (b * a + (1<<15)/2) / (1<<15);
      *dst_p++ = a;
    }
  }
}


void tile_rgba2flat(PyObject * dst_obj, PyObject * bg_obj) {
  PyArrayObject* bg = ((PyArrayObject*)bg_obj);
  PyArrayObject* dst = ((PyArrayObject*)dst_obj);

#ifdef HEAVY_DEBUG
  assert(PyArray_Check(dst_obj));
  assert(PyArray_DIM(dst, 0) == MYPAINT_TILE_SIZE);
  assert(PyArray_DIM(dst, 1) == MYPAINT_TILE_SIZE);
  assert(PyArray_DIM(dst, 2) == 4);
  assert(PyArray_TYPE(dst) == NPY_UINT16);
  assert(PyArray_ISCARRAY(dst));

  assert(PyArray_Check(bg_obj));
  assert(PyArray_DIM(bg, 0) == MYPAINT_TILE_SIZE);
  assert(PyArray_DIM(bg, 1) == MYPAINT_TILE_SIZE);
  assert(PyArray_DIM(bg, 2) == 4);
  assert(PyArray_TYPE(bg) == NPY_UINT16);
  assert(PyArray_ISCARRAY(bg));
#endif

  uint16_t * dst_p  = (uint16_t*)PyArray_DATA(dst);
  uint16_t * bg_p  = (uint16_t*)PyArray_DATA(bg);
  for (int i=0; i<MYPAINT_TILE_SIZE*MYPAINT_TILE_SIZE; i++) {
    // resultAlpha = 1.0 (thus it does not matter if resultColor is premultiplied alpha or not)
    // resultColor = topColor + (1.0 - topAlpha) * bottomColor
    const uint32_t one_minus_top_alpha = (1<<15) - dst_p[3];
    dst_p[0] += ((uint32_t)one_minus_top_alpha*bg_p[0]) / (1<<15);
    dst_p[1] += ((uint32_t)one_minus_top_alpha*bg_p[1]) / (1<<15);
    dst_p[2] += ((uint32_t)one_minus_top_alpha*bg_p[2]) / (1<<15);
    dst_p += 4;
    bg_p += 4;
  }
}


void tile_flat2rgba(PyObject * dst_obj, PyObject * bg_obj) {

  PyArrayObject *dst = (PyArrayObject *)dst_obj;
  PyArrayObject *bg = (PyArrayObject *)bg_obj;
#ifdef HEAVY_DEBUG
  assert(PyArray_Check(dst_obj));
  assert(PyArray_DIM(dst, 0) == MYPAINT_TILE_SIZE);
  assert(PyArray_DIM(dst, 1) == MYPAINT_TILE_SIZE);
  assert(PyArray_DIM(dst, 2) == 4);
  assert(PyArray_TYPE(dst) == NPY_UINT16);
  assert(PyArray_ISCARRAY(dst));

  assert(PyArray_Check(bg_obj));
  assert(PyArray_DIM(bg, 0) == MYPAINT_TILE_SIZE);
  assert(PyArray_DIM(bg, 1) == MYPAINT_TILE_SIZE);
  assert(PyArray_DIM(bg, 2) == 4);
  assert(PyArray_TYPE(bg) == NPY_UINT16);
  assert(PyArray_ISCARRAY(bg));
#endif

  uint16_t * dst_p  = (uint16_t*)PyArray_DATA(dst);
  uint16_t * bg_p  = (uint16_t*)PyArray_DATA(bg);
  for (int i=0; i<MYPAINT_TILE_SIZE*MYPAINT_TILE_SIZE; i++) {

    // 1. calculate final dst.alpha
    uint16_t final_alpha = dst_p[3];
    for (int i=0; i<3;i++) {
      int32_t color_change = (int32_t)dst_p[i] - bg_p[i];
      uint16_t minimal_alpha;
      if (color_change > 0) {
        minimal_alpha = (int64_t)color_change*(1<<15) / ((1<<15) - bg_p[i]);
      } else if (color_change < 0) {
        minimal_alpha = (int64_t)-color_change*(1<<15) / bg_p[i];
      } else {
        minimal_alpha = 0;
      }
      final_alpha = MAX(final_alpha, minimal_alpha);
#ifdef HEAVY_DEBUG
      assert(minimal_alpha <= (1<<15));
      assert(final_alpha   <= (1<<15));
#endif
    }

    // 2. calculate dst.color and update dst
    dst_p[3] = final_alpha;
    if (final_alpha > 0) {
      for (int i=0; i<3;i++) {
        int32_t color_change = (int32_t)dst_p[i] - bg_p[i];
        //int64_t res = bg_p[i] + (int64_t)color_change*(1<<15) / final_alpha;
        // premultiplied with final_alpha
        int64_t res = (uint32_t)bg_p[i]*final_alpha/(1<<15) + (int64_t)color_change;
        res = CLAMP(res, 0, final_alpha); // fix rounding errors
        dst_p[i] = res;
#ifdef HEAVY_DEBUG
        assert(dst_p[i] <= dst_p[3]);
#endif
      }
    } else {
      dst_p[0] = 0;
      dst_p[1] = 0;
      dst_p[2] = 0;
    }
    dst_p += 4;
    bg_p += 4;
  }
}


void tile_perceptual_change_strokemap(PyObject * a_obj, PyObject * b_obj, PyObject * res_obj) {

  PyArrayObject *a = (PyArrayObject *)a_obj;
  PyArrayObject *b = (PyArrayObject *)b_obj;
  PyArrayObject *res = (PyArrayObject *)res_obj;

#ifdef HEAVY_DEBUG
  assert(PyArray_TYPE(a) == NPY_UINT16);
  assert(PyArray_TYPE(b) == NPY_UINT16);
  assert(PyArray_TYPE(res) == NPY_UINT8);
  assert(PyArray_ISCARRAY(a));
  assert(PyArray_ISCARRAY(b));
  assert(PyArray_ISCARRAY(res));
#endif

  uint16_t * a_p  = (uint16_t*)PyArray_DATA(a);
  uint16_t * b_p  = (uint16_t*)PyArray_DATA(b);
  uint8_t * res_p = (uint8_t*)PyArray_DATA(res);

  for (int y=0; y<MYPAINT_TILE_SIZE; y++) {
    for (int x=0; x<MYPAINT_TILE_SIZE; x++) {

      int32_t color_change = 0;
      // We want to compare a.color with b.color, but we only know
      // (a.color * a.alpha) and (b.color * b.alpha).  We multiply
      // each component with the alpha of the other image, so they are
      // scaled the same and can be compared.

      for (int i=0; i<3; i++) {
        int32_t a_col = (uint32_t)a_p[i] * b_p[3] / (1<<15); // a.color * a.alpha*b.alpha
        int32_t b_col = (uint32_t)b_p[i] * a_p[3] / (1<<15); // b.color * a.alpha*b.alpha
        color_change += abs(b_col - a_col);
      }
      // "color_change" is in the range [0, 3*a_a]
      // if either old or new alpha is (near) zero, "color_change" is (near) zero

      int32_t alpha_old = a_p[3];
      int32_t alpha_new = b_p[3];

      // Note: the thresholds below are arbitrary choices found to work okay

      // We report a color change only if both old and new color are
      // well-defined (big enough alpha).
      bool is_perceptual_color_change = color_change > MAX(alpha_old, alpha_new)/16;

      int32_t alpha_diff = alpha_new - alpha_old; // no abs() here (ignore erasers)
      // We check the alpha increase relative to the previous alpha.
      bool is_perceptual_alpha_increase = alpha_diff > (1<<15)/4;

      // this one is responsible for making fat big ugly easy-to-hit pointer targets
      bool is_big_relative_alpha_increase  = alpha_diff > (1<<15)/64 && alpha_diff > alpha_old/2;

      if (is_perceptual_alpha_increase || is_big_relative_alpha_increase || is_perceptual_color_change) {
        res_p[0] = 1;
      } else {
        res_p[0] = 0;
      }

      a_p += 4;
      b_p += 4;
      res_p += 1;
    }
  }
}


// A named tile combine operation: what the user sees as a "blend mode" or 
// the "layer composite" modes in the application.

template <class B, class C>
class TileDataCombine : public TileDataCombineOp
{
  private:
    // The canonical name for the combine mode
    const char *name;
    // Alpha/nonalpha functors; must be members to keep GCC4.6 builds happy
    static const int bufsize = MYPAINT_TILE_SIZE*MYPAINT_TILE_SIZE*4;
    BufferCombineFunc<true, bufsize, B, C> combine_dstalpha;
    BufferCombineFunc<false, bufsize, B, C> combine_dstnoalpha;

  public:
    TileDataCombine(const char *name) {
        this->name = name;
    }

    // Apply this combine operation to source and destination tile-sized
    // buffers of uint16_t (15ish-bit) RGBA data. The output is written back
    // into the destination buffer.
    void combine_data (const fix15_short_t *src_p,
                       fix15_short_t *dst_p,
                       const bool dst_has_alpha,
                       const float src_opacity) const
    {
        const fix15_short_t opac = fix15_short_clamp(src_opacity * fix15_one);
        if (dst_has_alpha) {
            combine_dstalpha(src_p, dst_p, opac);
        }
        else {
            combine_dstnoalpha(src_p, dst_p, opac);
        }
    }

    // True if a zero-alpha source pixel can ever affect a destination pixel
    bool zero_alpha_has_effect() const {
        return C::zero_alpha_has_effect;
    }

    // True if a source pixel can ever reduce the alpha of a destination pixel
    bool can_decrease_alpha() const {
        return C::can_decrease_alpha;
    }

    // True if a zero-alpha src pixel always clears the dst pixel
    bool zero_alpha_clears_backdrop() const {
        return C::zero_alpha_clears_backdrop;
    }

    // Returns the canonical name of the mode
    const char* get_name() const {
        return name;
    }
};


// Integer-indexed LUT for the layer mode definitions, defining their canonical
// names.

static const TileDataCombineOp * combine_mode_info[NumCombineModes] =
{
    // Source-over compositing + various blend modes
    new TileDataCombine<BlendNormal, CompositeSourceOver>("svg:src-over"),
    new TileDataCombine<BlendMultiply, CompositeSourceOver>("svg:multiply"),
    new TileDataCombine<BlendScreen, CompositeSourceOver>("svg:screen"),
    new TileDataCombine<BlendOverlay, CompositeSourceOver>("svg:overlay"),
    new TileDataCombine<BlendDarken, CompositeSourceOver>("svg:darken"),
    new TileDataCombine<BlendLighten, CompositeSourceOver>("svg:lighten"),
    new TileDataCombine<BlendHardLight, CompositeSourceOver>("svg:hard-light"),
    new TileDataCombine<BlendSoftLight, CompositeSourceOver>("svg:soft-light"),
    new TileDataCombine<BlendColorBurn, CompositeSourceOver>("svg:color-burn"),
    new TileDataCombine<BlendColorDodge, CompositeSourceOver>("svg:color-dodge"),
    new TileDataCombine<BlendDifference, CompositeSourceOver>("svg:difference"),
    new TileDataCombine<BlendExclusion, CompositeSourceOver>("svg:exclusion"),
    new TileDataCombine<BlendHue, CompositeSourceOver>("svg:hue"),
    new TileDataCombine<BlendSaturation, CompositeSourceOver>("svg:saturation"),
    new TileDataCombine<BlendColor, CompositeSourceOver>("svg:color"),
    new TileDataCombine<BlendLuminosity, CompositeSourceOver>("svg:luminosity"),

    // Normal blend mode + various compositing operators
    new TileDataCombine<BlendNormal, CompositeLighter>("svg:plus"),
    new TileDataCombine<BlendNormal, CompositeDestinationIn>("svg:dst-in"),
    new TileDataCombine<BlendNormal, CompositeDestinationOut>("svg:dst-out"),
    new TileDataCombine<BlendNormal, CompositeSourceAtop>("svg:src-atop"),
    new TileDataCombine<BlendNormal, CompositeDestinationAtop>("svg:dst-atop"),
    new TileDataCombine<BlendNormal, CompositeSpectralWGM>("mypaint:spectral-wgm")
};



/* combine_mode_get_info(): extracts Python-readable metadata for a mode */


PyObject *
combine_mode_get_info(enum CombineMode mode)
{
    if (mode >= NumCombineModes || mode < 0) {
        return Py_BuildValue("{}");
    }
    const TileDataCombineOp *op = combine_mode_info[mode];
    return Py_BuildValue("{s:i,s:i,s:i,s:s}",
            "zero_alpha_has_effect", op->zero_alpha_has_effect(),
            "can_decrease_alpha", op->can_decrease_alpha(),
            "zero_alpha_clears_backdrop", op->zero_alpha_clears_backdrop(),
            "name", op->get_name()
        );
}



/* tile_combine(): primary Python interface for blending+compositing tiles */


void
tile_combine (enum CombineMode mode,
              PyObject *src_obj,
              PyObject *dst_obj,
              const bool dst_has_alpha,
              const float src_opacity)
{
    PyArrayObject* src = ((PyArrayObject*)src_obj);
    PyArrayObject* dst = ((PyArrayObject*)dst_obj);
#ifdef HEAVY_DEBUG
    assert(PyArray_Check(src_obj));
    assert(PyArray_DIM(src, 0) == MYPAINT_TILE_SIZE);
    assert(PyArray_DIM(src, 1) == MYPAINT_TILE_SIZE);
    assert(PyArray_DIM(src, 2) == 4);
    assert(PyArray_TYPE(src) == NPY_UINT16);
    assert(PyArray_ISCARRAY(src));

    assert(PyArray_Check(dst_obj));
    assert(PyArray_DIM(dst, 0) == MYPAINT_TILE_SIZE);
    assert(PyArray_DIM(dst, 1) == MYPAINT_TILE_SIZE);
    assert(PyArray_DIM(dst, 2) == 4);
    assert(PyArray_TYPE(dst) == NPY_UINT16);
    assert(PyArray_ISCARRAY(dst));

    assert(PyArray_STRIDES(dst)[0] == 4*sizeof(fix15_short_t)*MYPAINT_TILE_SIZE);
    assert(PyArray_STRIDES(dst)[1] == 4*sizeof(fix15_short_t));
    assert(PyArray_STRIDES(dst)[2] ==   sizeof(fix15_short_t));
#endif

    const fix15_short_t* const src_p = (fix15_short_t *)PyArray_DATA(src);
    fix15_short_t*       const dst_p = (fix15_short_t *)PyArray_DATA(dst);

    if (mode >= NumCombineModes || mode < 0) {
        return;
    }
    const TileDataCombineOp *op = combine_mode_info[mode];
    op->combine_data(src_p, dst_p, dst_has_alpha, src_opacity);
}

