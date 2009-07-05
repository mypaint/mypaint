/* This file is part of MyPaint.
 * Copyright (C) 2008-2009 by Martin Renold <martinxyz@gmx.ch>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */


void tile_composite_rgba16_over_rgb8(PyObject * src, PyObject * dst, float alpha) {
  /* disabled as optimization
  assert(PyArray_DIM(src, 0) == TILE_SIZE);
  assert(PyArray_DIM(src, 1) == TILE_SIZE);
  assert(PyArray_DIM(src, 2) == 4);
  assert(PyArray_TYPE(src) == NPY_UINT16);
  assert(PyArray_ISCARRAY(src));

  assert(PyArray_DIM(dst, 0) == TILE_SIZE);
  assert(PyArray_DIM(dst, 1) == TILE_SIZE);
  assert(PyArray_DIM(dst, 2) == 3);
  assert(PyArray_TYPE(dst) == NPY_UINT8);
  assert(PyArray_ISBEHAVED(dst));
  */
  
  PyArrayObject* dst_arr = ((PyArrayObject*)dst);
  /*
  assert(dst_arr->strides[1] == 3*sizeof(uint8_t));
  assert(dst_arr->strides[2] ==   sizeof(uint8_t));
  */

  uint32_t opac  = 255*alpha;
  uint32_t opac_ = opac*(1<<15)/255;
  uint16_t * src_p  = (uint16_t*)((PyArrayObject*)src)->data;
  char * p = dst_arr->data;
  for (int y=0; y<TILE_SIZE; y++) {
    uint8_t  * dst_p  = (uint8_t*) (p);
    for (int x=0; x<TILE_SIZE; x++) {
      // resultAlpha = 1.0 (thus it does not matter if resultColor is premultiplied alpha or not)
      // resultColor = topColor + (1.0 - topAlpha) * bottomColor
      const uint32_t one_minus_topAlpha = (1<<15) - (src_p[3]*opac_)/(1<<15);
      dst_p[0] = ((uint32_t)src_p[0]*opac + one_minus_topAlpha*dst_p[0]) / (1<<15);
      dst_p[1] = ((uint32_t)src_p[1]*opac + one_minus_topAlpha*dst_p[1]) / (1<<15);
      dst_p[2] = ((uint32_t)src_p[2]*opac + one_minus_topAlpha*dst_p[2]) / (1<<15);
      src_p += 4;
      dst_p += 3;
    }
    p += dst_arr->strides[0];
  }
}

// simply array copying (numpy assignment operator is about 13 times slower, sadly)
void tile_blit_rgb8_into_rgb8(PyObject * src, PyObject * dst) {
  PyArrayObject* src_arr = ((PyArrayObject*)src);
  PyArrayObject* dst_arr = ((PyArrayObject*)dst);

  /* disabled as optimization
  assert(PyArray_DIM(dst, 0) == TILE_SIZE);
  assert(PyArray_DIM(dst, 1) == TILE_SIZE);
  assert(PyArray_DIM(dst, 2) == 3);
  assert(PyArray_TYPE(dst) == NPY_UINT8);
  assert(PyArray_ISBEHAVED(dst));
  assert(dst_arr->strides[1] == 3*sizeof(uint8_t));
  assert(dst_arr->strides[2] ==   sizeof(uint8_t));

  assert(PyArray_DIM(src, 0) == TILE_SIZE);
  assert(PyArray_DIM(src, 1) == TILE_SIZE);
  assert(PyArray_DIM(src, 2) == 3);
  assert(PyArray_TYPE(src) == NPY_UINT8);
  assert(PyArray_ISBEHAVED(src));
  assert(src_arr->strides[1] == 3*sizeof(uint8_t));
  assert(src_arr->strides[2] ==   sizeof(uint8_t));
  */

  char * src_p = src_arr->data;
  char * dst_p = dst_arr->data;
  for (int y=0; y<TILE_SIZE; y++) {
    memcpy(dst_p, src_p, TILE_SIZE*3);
    src_p += src_arr->strides[0];
    dst_p += dst_arr->strides[0];
  }
}

// used mainly for saving layers (transparent PNG)
void tile_convert_rgba16_to_rgba8(PyObject * src, PyObject * dst) {
  PyArrayObject* src_arr = ((PyArrayObject*)src);
  PyArrayObject* dst_arr = ((PyArrayObject*)dst);

  /* disabled as optimization
  assert(PyArray_DIM(dst, 0) == TILE_SIZE);
  assert(PyArray_DIM(dst, 1) == TILE_SIZE);
  assert(PyArray_DIM(dst, 2) == 4);
  assert(PyArray_TYPE(dst) == NPY_UINT8);
  assert(PyArray_ISBEHAVED(dst));
  assert(dst_arr->strides[1] == 4*sizeof(uint8_t));
  assert(dst_arr->strides[2] ==   sizeof(uint8_t));

  assert(PyArray_DIM(src, 0) == TILE_SIZE);
  assert(PyArray_DIM(src, 1) == TILE_SIZE);
  assert(PyArray_DIM(src, 2) == 4);
  assert(PyArray_TYPE(src) == NPY_UINT16);
  assert(PyArray_ISBEHAVED(src));
  assert(src_arr->strides[1] == 4*sizeof(uint16_t));
  assert(src_arr->strides[2] ==   sizeof(uint16_t));
  */

  for (int y=0; y<TILE_SIZE; y++) {
    uint16_t * src_p = (uint16_t*)(src_arr->data + y*src_arr->strides[0]);
    uint8_t  * dst_p = (uint8_t*)(dst_arr->data + y*dst_arr->strides[0]);
    for (int x=0; x<TILE_SIZE; x++) {
      uint32_t r, g, b, a;
      r = *src_p++;
      g = *src_p++;
      b = *src_p++;
      a = *src_p++;
      /*
      assert(a<=(1<<15));
      assert(r<=(1<<15));
      assert(g<=(1<<15));
      assert(b<=(1<<15));
      assert(r<=a);
      assert(g<=a);
      assert(b<=a);
      */
      // un-premultiply alpha (with rounding)
      if (a != 0) {
        r = ((r << 15) + a/2) / a;
        g = ((g << 15) + a/2) / a;
        b = ((b << 15) + a/2) / a;
      } else {
        r = g = b = 0;
      }
      /*
      assert(a<=(1<<15));
      assert(r<=(1<<15));
      assert(g<=(1<<15));
      assert(b<=(1<<15));
      */

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
      const uint32_t add_r = random() % (1<<15);
      const uint32_t add_g = random() % (1<<15);
      const uint32_t add_b = random() % (1<<15);
      const uint32_t add_a = random() % (1<<15);
      */

      // Variant C) slightly better dithering
      // make sure we don't dither rounding errors (those did occur when converting 8bit-->16bit)
      // this preserves the alpha channel, but we still add noise to the highly transparent colors
      // OPTIMIZE: calling random() slows us down...
      const uint32_t add_r = (random() % (1<<15)) * 240/256 + (1<<15) * 8/256;
      const uint32_t add_g = add_r; // hm... do not produce too much color noise
      const uint32_t add_b = add_r;
      const uint32_t add_a = (random() % (1<<15)) * 240/256 + (1<<15) * 8/256;
      // TODO: error diffusion might work better than random dithering...

      /*
      assert(add_a < (1<<15));
      assert(add_a >= 0);
      */

      *dst_p++ = (r * 255 + add_r) / (1<<15);
      *dst_p++ = (g * 255 + add_g) / (1<<15);
      *dst_p++ = (b * 255 + add_b) / (1<<15);
      *dst_p++ = (a * 255 + add_a) / (1<<15);
    }
    src_p += src_arr->strides[0];
    dst_p += dst_arr->strides[0];
  }
}

// used mainly for loading layers (transparent PNG)
void tile_convert_rgba8_to_rgba16(PyObject * src, PyObject * dst) {
  PyArrayObject* src_arr = ((PyArrayObject*)src);
  PyArrayObject* dst_arr = ((PyArrayObject*)dst);

  /* disabled as optimization
  assert(PyArray_DIM(dst, 0) == TILE_SIZE);
  assert(PyArray_DIM(dst, 1) == TILE_SIZE);
  assert(PyArray_DIM(dst, 2) == 4);
  assert(PyArray_TYPE(dst) == NPY_UINT16);
  assert(PyArray_ISBEHAVED(dst));
  assert(dst_arr->strides[1] == 4*sizeof(uint16_t));
  assert(dst_arr->strides[2] ==   sizeof(uint16_t));

  assert(PyArray_DIM(src, 0) == TILE_SIZE);
  assert(PyArray_DIM(src, 1) == TILE_SIZE);
  assert(PyArray_DIM(src, 2) == 4);
  assert(PyArray_TYPE(src) == NPY_UINT8);
  assert(PyArray_ISBEHAVED(src));
  assert(src_arr->strides[1] == 4*sizeof(uint8_t));
  assert(src_arr->strides[2] ==   sizeof(uint8_t));
  */

  for (int y=0; y<TILE_SIZE; y++) {
    uint8_t  * src_p = (uint8_t*)(src_arr->data + y*src_arr->strides[0]);
    uint16_t * dst_p = (uint16_t*)(dst_arr->data + y*dst_arr->strides[0]);
    for (int x=0; x<TILE_SIZE; x++) {
      uint32_t r, g, b, a;
      r = *src_p++;
      g = *src_p++;
      b = *src_p++;
      a = *src_p++;

      // convert to fixed point (with rounding)
      r = (r * (1<<15) + 255/2) / 255;
      g = (g * (1<<15) + 255/2) / 255;
      b = (b * (1<<15) + 255/2) / 255;
      a = (a * (1<<15) + 255/2) / 255;

      // premultiply alpha (with rounding), save back
      *dst_p++ = (r * a + (1<<15)/2) / (1<<15);
      *dst_p++ = (g * a + (1<<15)/2) / (1<<15);
      *dst_p++ = (b * a + (1<<15)/2) / (1<<15);
      *dst_p++ = a;
    }
  }
}
