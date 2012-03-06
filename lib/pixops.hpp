/* This file is part of MyPaint.
 * Copyright (C) 2008-2009 by Martin Renold <martinxyz@gmx.ch>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */

// make the "heavy_debug" readable from python
#ifdef HEAVY_DEBUG
const bool heavy_debug = true;
#else
const bool heavy_debug = false;
#endif

// downscale a tile to half its size using bilinear interpolation
// used for generating mipmaps for tiledsurface and background
void tile_downscale_rgba16(PyObject *src, PyObject *dst, int dst_x, int dst_y) {
#ifdef HEAVY_DEBUG
  assert(PyArray_DIM(src, 0) == TILE_SIZE);
  assert(PyArray_DIM(src, 1) == TILE_SIZE);
  assert(PyArray_DIM(src, 2) == 4);
  assert(PyArray_TYPE(src) == NPY_UINT16);
  assert(PyArray_ISCARRAY(src));

  assert(PyArray_DIM(dst, 2) == 4);
  assert(PyArray_TYPE(dst) == NPY_UINT16);
  assert(PyArray_ISCARRAY(dst));
#endif

  PyArrayObject* src_arr = ((PyArrayObject*)src);
  PyArrayObject* dst_arr = ((PyArrayObject*)dst);

  for (int y=0; y<TILE_SIZE/2; y++) {
    uint16_t * src_p = (uint16_t*)(src_arr->data + (2*y)*src_arr->strides[0]);
    uint16_t * dst_p = (uint16_t*)(dst_arr->data + (y+dst_y)*dst_arr->strides[0]);
    dst_p += 4*dst_x;
    for(int x=0; x<TILE_SIZE/2; x++) {
      dst_p[0] = src_p[0]/4 + (src_p+4)[0]/4 + (src_p+4*TILE_SIZE)[0]/4 + (src_p+4*TILE_SIZE+4)[0]/4;
      dst_p[1] = src_p[1]/4 + (src_p+4)[1]/4 + (src_p+4*TILE_SIZE)[1]/4 + (src_p+4*TILE_SIZE+4)[1]/4;
      dst_p[2] = src_p[2]/4 + (src_p+4)[2]/4 + (src_p+4*TILE_SIZE)[2]/4 + (src_p+4*TILE_SIZE+4)[2]/4;
      dst_p[3] = src_p[3]/4 + (src_p+4)[3]/4 + (src_p+4*TILE_SIZE)[3]/4 + (src_p+4*TILE_SIZE+4)[3]/4;
      src_p += 8;
      dst_p += 4;
    }
  }
}


/**
 * tile_composite_src_over:
 *
 * @src: upper source tile, unmodified
 * @dst: lower destination tile, will be modified
 * @dst_has_alpha: true if @dst's alpha should be processed
 * @src_opacity: overall multiplier for @src's alpha
 *
 * The default layer compositing operation. Composites two tiles using the
 * usual Porter-Duff OVER operator, src OVER dst.
 *
 * Dimensions of both arrays must be (TILE_SIZE, TILE_SIZE, 4). If
 * @dst_has_alpha is false, @dst's alpha is ignored and treated as 100%, which
 * results in faster operation and generates opaque output.
 */

void
tile_composite_src_over (PyObject *src,
                         PyObject *dst,
                         const bool dst_has_alpha,
                         const float src_opacity)
{
#ifdef HEAVY_DEBUG
  assert(PyArray_DIM(src, 0) == TILE_SIZE);
  assert(PyArray_DIM(src, 1) == TILE_SIZE);
  assert(PyArray_DIM(src, 2) == 4);
  assert(PyArray_TYPE(src) == NPY_UINT16);
  assert(PyArray_ISCARRAY(src));

  assert(PyArray_DIM(dst, 0) == TILE_SIZE);
  assert(PyArray_DIM(dst, 1) == TILE_SIZE);
  assert(PyArray_DIM(dst, 2) == 4);
  assert(PyArray_TYPE(dst) == NPY_UINT16);
  assert(PyArray_ISBEHAVED(dst));
#endif

  PyArrayObject* dst_arr = ((PyArrayObject*)dst);
#ifdef HEAVY_DEBUG
  assert(dst_arr->strides[1] == 4*sizeof(uint16_t));
  assert(dst_arr->strides[2] ==   sizeof(uint16_t));
#endif

  // The overall layer opacity is applied as a scaling factor to the src
  // layer's alpha, and to the premultiplied src colour components. In the
  // derivations below,
  //
  // Sa = opac * src_p[3]       ; src's effective alpha
  // Sca = opac * src_p[c]      ; effective src component incl. premult alpha

  const uint32_t opac = CLAMP(src_opacity * (1<<15) + 0.5, 0, (1<<15));
  if (opac == 0) return;

  uint16_t *src_p  = (uint16_t*)((PyArrayObject*)src)->data;
  char *p = dst_arr->data;
  for (int y=0; y<TILE_SIZE; y++) {
    uint16_t  *dst_p  = (uint16_t*) (p);
    for (int x=0; x<TILE_SIZE; x++) {
      const uint16_t src_alpha = CLAMP((uint32_t)(opac*src_p[3])>>15, 0, 1<<15);
      const uint32_t one_minus_src_alpha = (1<<15) - src_alpha;

      // Dca: destination component with premult alpha
      // Da: destination alpha channel
      // 
      // Dca' = Sca*Da + Sca*(1 - Da) + Dca*(1 - Sa)
      //      = Sca + Dca*(1 - Sa)
      //
      dst_p[0] = ((uint32_t)src_p[0]*opac + one_minus_src_alpha*dst_p[0]) >> 15;
      dst_p[1] = ((uint32_t)src_p[1]*opac + one_minus_src_alpha*dst_p[1]) >> 15;
      dst_p[2] = ((uint32_t)src_p[2]*opac + one_minus_src_alpha*dst_p[2]) >> 15;

      // Da' = Sa*Da + Sa*(1 - Da) + Da*(1 - Sa)
      //     = (Sa*Da + Sa - Sa*Da) + Da*(1 - Sa)
      //     = Sa + Da*(1 - Sa)
      //
      if (dst_has_alpha) {
        dst_p[3] = src_alpha + ((one_minus_src_alpha*dst_p[3]) / (1<<15));
      }
      src_p += 4;
      dst_p += 4;
    }
    p += dst_arr->strides[0];
  }
}


/**
 * tile_composite_multiply:
 *
 * @src: upper source tile, unmodified
 * @dst: lower destination tile, will be modified
 * @dst_has_alpha: true if @dst's alpha should be processed
 * @src_opacity: overall multiplier for @src's alpha
 *
 * Multiplies together each pixel in @src and @dst, writing the result into
 * @dst. The result is always at least as dark as either of the input tiles.
 *
 * Dimensions of both arrays must be (TILE_SIZE, TILE_SIZE, 4). If
 * @dst_has_alpha is false, @dst's alpha is ignored and treated as 100%, which
 * results in faster operation and generates opaque output.
 */

void
tile_composite_multiply (PyObject *src,
                         PyObject *dst,
                         const bool dst_has_alpha,
                         const float src_opacity)
{
#ifdef HEAVY_DEBUG
  assert(PyArray_DIM(src, 0) == TILE_SIZE);
  assert(PyArray_DIM(src, 1) == TILE_SIZE);
  assert(PyArray_DIM(src, 2) == 4);
  assert(PyArray_TYPE(src) == NPY_UINT16);
  assert(PyArray_ISCARRAY(src));

  assert(PyArray_DIM(dst, 0) == TILE_SIZE);
  assert(PyArray_DIM(dst, 1) == TILE_SIZE);
  assert(PyArray_DIM(dst, 2) == 4);
  assert(PyArray_TYPE(dst) == NPY_UINT16);
  assert(PyArray_ISBEHAVED(dst));
#endif

  PyArrayObject* dst_arr = ((PyArrayObject*)dst);
#ifdef HEAVY_DEBUG
  assert(dst_arr->strides[1] == 4*sizeof(uint16_t));
  assert(dst_arr->strides[2] ==   sizeof(uint16_t));
#endif

  const uint32_t opac = CLAMP(src_opacity * (1<<15) + 0.5, 0, (1<<15));
  if (opac == 0) return;

  uint16_t *src_p  = (uint16_t*)((PyArrayObject*)src)->data;
  char *p = dst_arr->data;
  for (int y=0; y<TILE_SIZE; y++) {
    uint16_t  * dst_p  = (uint16_t*) (p);
    for (int x=0; x<TILE_SIZE; x++) {

      // Dca' = Sca*Dca + Sca*(1 - Da) + Dca*(1 - Sa)
      //
      // If Da == 1, which is the case without dst_has_alpha, this becomes
      //
      // Dca' = Sca*Dca + 0 + Dca*(1 - Sa)
      //      = Dca * (Sca + (1 - Sa))
      //
      const uint16_t src_alpha = CLAMP((uint32_t)(opac * src_p[3]) / (1<<15),
                                       0, 1<<15);
      const uint32_t one_minus_src_alpha = (1<<15) - src_alpha;
      const uint32_t src_col0 = ((uint32_t) src_p[0] * opac) >> 15;
      const uint32_t src_col1 = ((uint32_t) src_p[1] * opac) >> 15;
      const uint32_t src_col2 = ((uint32_t) src_p[2] * opac) >> 15;
      dst_p[0] = ((uint32_t)src_col0*dst_p[0] + one_minus_src_alpha*dst_p[0])
                 / (1<<15);
      dst_p[1] = ((uint32_t)src_col1*dst_p[1] + one_minus_src_alpha*dst_p[1])
                 / (1<<15);
      dst_p[2] = ((uint32_t)src_col2*dst_p[2] + one_minus_src_alpha*dst_p[2])
                 / (1<<15);
      if (dst_has_alpha) {
        // Sca*(1 - Da) != 0, add it in
        const uint32_t one_minus_dst_alpha = (1<<15) - dst_p[3];
        dst_p[0] += (((uint32_t)src_col0 * one_minus_dst_alpha) >> 15);
        dst_p[1] += (((uint32_t)src_col1 * one_minus_dst_alpha) >> 15);
        dst_p[2] += (((uint32_t)src_col2 * one_minus_dst_alpha) >> 15);

        // Da'  = Sa*Da + Sa*(1 - Da) + Da*(1 - Sa)
        //      = (Sa*Da + Sa - Sa*Da) + Da*(1 - Sa)
        //      = Sa + (1 - Sa)*Da
        dst_p[3] = src_alpha + ((one_minus_src_alpha*dst_p[3]) / (1<<15));
      }
      src_p += 4;
      dst_p += 4;
    }
    p += dst_arr->strides[0];
  }
}


/**
 * tile_composite_screen:
 *
 * @src: upper source tile, unmodified
 * @dst: lower destination tile, will be modified
 * @dst_has_alpha: true if @dst's alpha should be processed
 * @src_opacity: overall multiplier for @src's alpha
 *
 * Multiplies together the complements of each pixel in @src and @dst, writing
 * the result into @dst. The result is always lighter than either of the input
 * tiles.
 *
 * Dimensions of both arrays must be (TILE_SIZE, TILE_SIZE, 4). If
 * @dst_has_alpha is false, @dst's alpha is ignored and treated as 100%, which
 * results in faster operation and generates opaque output.
 */

void
tile_composite_screen (PyObject *src,
                       PyObject *dst,
                       const bool dst_has_alpha,
                       const float src_opacity)
{
#ifdef HEAVY_DEBUG
  assert(PyArray_DIM(src, 0) == TILE_SIZE);
  assert(PyArray_DIM(src, 1) == TILE_SIZE);
  assert(PyArray_DIM(src, 2) == 4);
  assert(PyArray_TYPE(src) == NPY_UINT16);
  assert(PyArray_ISCARRAY(src));

  assert(PyArray_DIM(dst, 0) == TILE_SIZE);
  assert(PyArray_DIM(dst, 1) == TILE_SIZE);
  assert(PyArray_DIM(dst, 2) == 4);
  assert(PyArray_TYPE(dst) == NPY_UINT16);
  assert(PyArray_ISBEHAVED(dst));
#endif
  PyArrayObject* dst_arr = ((PyArrayObject*)dst);
#ifdef HEAVY_DEBUG
  assert(dst_arr->strides[1] == 4*sizeof(uint16_t));
  assert(dst_arr->strides[2] ==   sizeof(uint16_t));
#endif

  const uint32_t opac = CLAMP(src_opacity * (1<<15) + 0.5, 0, (1<<15));
  if (opac == 0) return;

  uint16_t *src_p  = (uint16_t*)((PyArrayObject*)src)->data;
  char *p = dst_arr->data;
  for (int y=0; y<TILE_SIZE; y++) {
    uint16_t  * dst_p  = (uint16_t*) (p);
    for (int x=0; x<TILE_SIZE; x++) {
      // Dca' = (Sca*Da + Dca*Sa - Sca*Dca) + Sca*(1 - Da) + Dca*(1 - Sa)
      //      = Sca + Dca - Sca*Dca
      const uint32_t col0 = ((uint32_t)src_p[0]*opac)
                            + (((uint32_t)dst_p[0]) << 15);
      const uint32_t col1 = ((uint32_t)src_p[1]*opac)
                            + (((uint32_t)dst_p[1]) << 15);
      const uint32_t col2 = ((uint32_t)src_p[2]*opac)
                            + (((uint32_t)dst_p[2]) << 15);
      const uint32_t src_col0 = ((uint32_t)src_p[0] * opac) >> 15;
      const uint32_t src_col1 = ((uint32_t)src_p[1] * opac) >> 15;
      const uint32_t src_col2 = ((uint32_t)src_p[2] * opac) >> 15;
      dst_p[0] = (col0 - ((uint32_t)src_col0*dst_p[0])) / (1<<15);
      dst_p[1] = (col1 - ((uint32_t)src_col1*dst_p[1])) / (1<<15);
      dst_p[2] = (col2 - ((uint32_t)src_col2*dst_p[2])) / (1<<15);
      if (dst_has_alpha) {
        // Da'  = Sa + Da - Sa*Da
        const uint32_t src_alpha = ((uint32_t)src_p[3] * opac) >> 15;
        dst_p[3] =   src_alpha + (uint32_t)dst_p[3]
                 - ((src_alpha * (uint32_t)dst_p[3]) >> 15);
      }
      src_p += 4;
      dst_p += 4;
    }
    p += dst_arr->strides[0];
  }
}



/**
 * tile_composite_color_dodge:
 *
 * @src: upper source tile, unmodified
 * @dst: lower destination tile, will be modified
 * @dst_has_alpha: true if @dst's alpha should be processed
 * @src_opacity: overall multiplier for @src's alpha
 *
 * Brightens @dst to reflect @src. Using black in @src preserves the colour in
 * @dst.
 *
 * Dimensions of both arrays must be (TILE_SIZE, TILE_SIZE, 4). If
 * @dst_has_alpha is false, @dst's alpha is ignored and treated as 100%, which
 * results in faster operation and generates opaque output.
 */

void
tile_composite_color_dodge (PyObject *src,
                            PyObject *dst,
                            const bool dst_has_alpha,
                            const float src_opacity)
{
#ifdef HEAVY_DEBUG
  assert(PyArray_DIM(src, 0) == TILE_SIZE);
  assert(PyArray_DIM(src, 1) == TILE_SIZE);
  assert(PyArray_DIM(src, 2) == 4);
  assert(PyArray_TYPE(src) == NPY_UINT16);
  assert(PyArray_ISCARRAY(src));

  assert(PyArray_DIM(dst, 0) == TILE_SIZE);
  assert(PyArray_DIM(dst, 1) == TILE_SIZE);
  assert(PyArray_DIM(dst, 2) == 4);
  assert(PyArray_TYPE(dst) == NPY_UINT16);
  assert(PyArray_ISBEHAVED(dst));
#endif

  PyArrayObject* dst_arr = ((PyArrayObject*)dst);
#ifdef HEAVY_DEBUG
  assert(dst_arr->strides[1] == 4*sizeof(uint16_t));
  assert(dst_arr->strides[2] ==   sizeof(uint16_t));
#endif

  const uint32_t opac = CLAMP(src_opacity * (1<<15) + 0.5, 0, (1<<15));
  if (opac == 0) return;

  uint16_t *src_p  = (uint16_t*)((PyArrayObject*)src)->data;
  char *p = dst_arr->data;
  for (int y=0; y<TILE_SIZE; y++) {
    uint16_t *dst_p  = (uint16_t*) (p);
    for (int x=0; x<TILE_SIZE; x++) {
      const uint32_t src_alpha = ((uint32_t)src_p[3]*opac)>>15;
      const uint32_t one_minus_src_alpha = (1<<15) - src_alpha;
      const uint32_t dst_alpha = dst_has_alpha ? dst_p[3] : (1<<15);
      const uint32_t one_minus_dst_alpha = (1<<15) - dst_alpha;

      for (int c=0; c < 3; c++) {
        const uint32_t s = ((uint32_t)src_p[c]*opac)>>15;
        const uint32_t d = dst_p[c];
        const uint32_t src_alpha_minus_src = src_alpha - s;
        if (src_alpha_minus_src == 0 && d == 0) {
          // Sca == Sa and Dca == 0
          //  Dca' = Sca*(1 - Da) + Dca*(1 - Sa)
          //       = Sca*(1 - Da)
          dst_p[c] = CLAMP((s * one_minus_dst_alpha) >> 15, 0, (1<<15));
        }
        else if (src_alpha_minus_src == 0) {
          // otherwise if Sca == Sa
          //  Dca' = Sa*Da + Sca*(1 - Da) + Dca*(1 - Sa)
          //       = Sca*Da + Sca*(1 - Da) + Dca*(1 - Sa)
          //       = Sca*(Da + 1 - Da) + Dca*(1 - Sa)
          //       = Sca + Dca*(1 - Sa)
          dst_p[c] = CLAMP( s + ((d * one_minus_src_alpha)>>15),
                            0, (1<<15) );
        }
        else {
          // Sca < Sa
          //    Dca' = Sa*Da * min(1, Dca/Da * Sa/(Sa - Sca))
          //          + Sca*(1 - Da) + Dca*(1 - Sa)
          const uint32_t dst_times_src_alpha_B30 = d*src_alpha;
          // when 1 < Dca/Da * Sa/(Sa - Sca) 
          //      1 < (Dca*Sa) / (Da*(Sa - Sca)
          //  (Da*(Sa - Sca) < (Dca*Sa)   because Sca - Sa is -ve and nonzero
          if (dst_times_src_alpha_B30 > (dst_alpha * src_alpha_minus_src)) {
            // min(...)==1
            //    Dca' = Sa * Da * min(...) + Sca*(1 - Da) + Dca*(1 - Sa)
            //    Dca' = Sa * Da + Sca*(1 - Da) + Dca*(1 - Sa)
            dst_p[c] = CLAMP(
                    (  (src_alpha * dst_alpha)   // B30
                     + (s * one_minus_dst_alpha) // B30
                     + (d * one_minus_src_alpha) // B30
                    ) >> 15,
                0, 1<<15);
          }
          else {
            // min(...) == Dca/Da * Sa/(Sa - Sca)
            //    Dca' = Sa * Da * min(...) + Sca*(1 - Da) + Dca*(1 - Sa)
            //    Dca' = Sa * Da * Dca/Da * Sa/(Sa - Sca)
            //            + Sca*(1 - Da) + Dca*(1 - Sa)
            //    Dca' = Sa * Dca * Sa/(Sa - Sca)
            //            + Sca*(1 - Da) + Dca*(1 - Sa)
            dst_p[c] = CLAMP(
                     ( src_alpha * (dst_times_src_alpha_B30>>15)
                       / src_alpha_minus_src )
                     + ((s * one_minus_dst_alpha) >> 15)
                     + ((d * one_minus_src_alpha) >> 15),
                 0, 1<<15);
          }
        }
#ifdef HEAVY_DEBUG
        assert(dst_p[c] <= (1<<15));
        assert(src_p[c] <= (1<<15));
#endif
      }
      if (dst_has_alpha) {
         // Da'  = Sa + Da - Sa*Da
         dst_p[3] = CLAMP(src_alpha + dst_alpha - ((src_alpha*dst_alpha)>>15),
                          0, (1<<15));
#ifdef HEAVY_DEBUG
         assert(src_p[0] <= src_p[3]);
         assert(dst_p[0] <= dst_p[3]);
         assert(src_p[1] <= src_p[3]);
         assert(dst_p[1] <= dst_p[3]);
         assert(src_p[2] <= src_p[3]);
         assert(dst_p[2] <= dst_p[3]);
#endif
      }
#ifdef HEAVY_DEBUG
      assert(dst_p[3] <= (1<<15));
      assert(src_p[3] <= (1<<15));
#endif
      src_p += 4;
      dst_p += 4;
    }

    p += dst_arr->strides[0];
  }
}



/**
 * tile_composite_color_burn:
 *
 * @src: upper source tile, unmodified
 * @dst: lower destination tile, will be modified
 * @dst_has_alpha: true if @dst's alpha should be processed
 * @src_opacity: overall multiplier for @src's alpha
 *
 * Darkens @dst to reflect @src. Using white in @src preserves the colour in
 * @dst.
 *
 * Dimensions of both arrays must be (TILE_SIZE, TILE_SIZE, 4). If
 * @dst_has_alpha is false, @dst's alpha is ignored and treated as 100%, which
 * results in faster operation and generates opaque output.
 */

void
tile_composite_color_burn (PyObject *src,
                           PyObject *dst,
                           const bool dst_has_alpha,
                           const float src_opacity)
{
#ifdef HEAVY_DEBUG
  assert(PyArray_DIM(src, 0) == TILE_SIZE);
  assert(PyArray_DIM(src, 1) == TILE_SIZE);
  assert(PyArray_DIM(src, 2) == 4);
  assert(PyArray_TYPE(src) == NPY_UINT16);
  assert(PyArray_ISCARRAY(src));

  assert(PyArray_DIM(dst, 0) == TILE_SIZE);
  assert(PyArray_DIM(dst, 1) == TILE_SIZE);
  assert(PyArray_DIM(dst, 2) == 4);
  assert(PyArray_TYPE(dst) == NPY_UINT16);
  assert(PyArray_ISBEHAVED(dst));
#endif

  PyArrayObject* dst_arr = ((PyArrayObject*)dst);
#ifdef HEAVY_DEBUG
  assert(dst_arr->strides[1] == 4*sizeof(uint16_t));
  assert(dst_arr->strides[2] ==   sizeof(uint16_t));
#endif

  const uint32_t opac = CLAMP(src_opacity * (1<<15) + 0.5, 0, (1<<15));
  if (opac == 0) return;

  uint16_t * src_p  = (uint16_t*)((PyArrayObject*)src)->data;
  char * p = dst_arr->data;
  for (int y=0; y<TILE_SIZE; y++) {
    uint16_t  * dst_p  = (uint16_t*) (p);
    for (int x=0; x<TILE_SIZE; x++) {
      const uint32_t src_alpha30 = (uint32_t)src_p[3]*opac;
      const uint32_t src_alpha = src_alpha30>>15;
      const uint32_t one_minus_src_alpha = (1<<15) - src_alpha;
      const uint32_t dst_alpha = dst_has_alpha ? dst_p[3] : (1<<15);
      const uint32_t one_minus_dst_alpha = (1<<15) - dst_alpha;

      for (int c=0; c<3; c++) {
        const uint32_t s30 = (uint32_t)src_p[c] * opac;
        const uint32_t s = s30 >> 15;
        const uint32_t d = dst_p[c];
        if (s == 0) {
          if (d != dst_alpha) {
            //if Sca == 0 and Dca == Da
            //  Dca' = Sa*Da + Sca*(1 - Da) + Dca*(1 - Sa)
            //       = Sa*Dca + Dca*(1 - Sa)
            //       = Sa*Dca + Dca - Sa*Dca
            //       = Dca

            //otherwise if Sca == 0
            //  Dca' = Sca*(1 - Da) + Dca*(1 - Sa)
            //       = Dca*(1 - Sa)
            dst_p[c] = CLAMP(((d * one_minus_src_alpha) >> 15), 0, (1<<15));
          }
        }
        else {
#ifdef HEAVY_DEBUG
          assert(s <= (1<<15));
          assert(s > 0);
#endif
          //otherwise if Sca > 0
          //  let i = Sca*(1 - Da) + Dca*(1 - Sa)
          //  let m = (1 - Dca/Da) * Sa/Sca
          //
          //  Dca' = Sa*Da - Sa*Da * min(1, (1 - Dca/Da) * Sa/Sca) + i
          //       = Sa*Da * (1 - min(1, (1 - Dca/Da) * Sa/Sca)) + i

          uint32_t res = (s*one_minus_dst_alpha + d*one_minus_src_alpha)>>15;
          if (dst_alpha > 0) {
            const uint32_t m = (  ((1<<15) - ((d << 15) / dst_alpha))
                                * src_alpha) / s;
            if (m < (1<<15)) {
              res += (  ((src_alpha * dst_alpha) >> 15)
                      * ((1<<15) - m) ) >> 15;
            }
          }
          dst_p[c] = CLAMP(res, 0, (1<<15));
        }
#ifdef HEAVY_DEBUG
        assert(dst_p[c] <= (1<<15));
        assert(src_p[c] <= (1<<15));
#endif
      }
      if (dst_has_alpha) {
         // Da'  = Sa + Da - Sa*Da
         dst_p[3] = CLAMP(src_alpha + dst_alpha - ((src_alpha*dst_alpha)>>15),
                          0, (1<<15));
#ifdef HEAVY_DEBUG
         assert(src_p[0] <= src_p[3]);
         assert(dst_p[0] <= dst_p[3]);
         assert(src_p[1] <= src_p[3]);
         assert(dst_p[1] <= dst_p[3]);
         assert(src_p[2] <= src_p[3]);
         assert(dst_p[2] <= dst_p[3]);
#endif
      }
#ifdef HEAVY_DEBUG
      assert(dst_p[3] <= (1<<15));
      assert(src_p[3] <= (1<<15));
#endif
      src_p += 4;
      dst_p += 4;
    }
    p += dst_arr->strides[0];
  }
}

// used to e.g. copy the background before starting to composite over it
//
// simply array copying (numpy assignment operator) is about 13 times slower, sadly
// The above comment is true when the array is sliced; it's only about two
// times faster now, in the current usecae.
void tile_copy_rgba16_into_rgba16(PyObject * src, PyObject * dst) {
  PyArrayObject* src_arr = ((PyArrayObject*)src);
  PyArrayObject* dst_arr = ((PyArrayObject*)dst);

#ifdef HEAVY_DEBUG
  assert(PyArray_DIM(dst, 0) == TILE_SIZE);
  assert(PyArray_DIM(dst, 1) == TILE_SIZE);
  assert(PyArray_DIM(dst, 2) == 4);
  assert(PyArray_TYPE(dst) == NPY_UINT16);
  assert(PyArray_ISCARRAY(dst));
  assert(dst_arr->strides[1] == 4*sizeof(uint16_t));
  assert(dst_arr->strides[2] ==   sizeof(uint16_t));

  assert(PyArray_DIM(src, 0) == TILE_SIZE);
  assert(PyArray_DIM(src, 1) == TILE_SIZE);
  assert(PyArray_DIM(src, 2) == 4);
  assert(PyArray_TYPE(src) == NPY_UINT16);
  assert(PyArray_ISCARRAY(dst));
  assert(src_arr->strides[1] == 4*sizeof(uint16_t));
  assert(src_arr->strides[2] ==   sizeof(uint16_t));
#endif

  memcpy(dst_arr->data, src_arr->data, TILE_SIZE*TILE_SIZE*4*sizeof(uint16_t));
  /* the code below can be used if it is not ISCARRAY, but only ISBEHAVED:
  char * src_p = src_arr->data;
  char * dst_p = dst_arr->data;
  for (int y=0; y<TILE_SIZE; y++) {
    memcpy(dst_p, src_p, TILE_SIZE*4);
    src_p += src_arr->strides[0];
    dst_p += dst_arr->strides[0];
  }
  */
}

void tile_clear(PyObject * dst) {
  PyArrayObject* dst_arr = ((PyArrayObject*)dst);

#ifdef HEAVY_DEBUG
  assert(PyArray_DIM(dst, 0) == TILE_SIZE);
  assert(PyArray_DIM(dst, 1) == TILE_SIZE);
  assert(PyArray_TYPE(dst) == NPY_UINT8);
  assert(PyArray_ISBEHAVED(dst));
  assert(dst_arr->strides[1] <= 8);
#endif

  for (int y=0; y<TILE_SIZE; y++) {
    uint8_t  * dst_p = (uint8_t*)(dst_arr->data + y*dst_arr->strides[0]);
    memset(dst_p, 0, TILE_SIZE*dst_arr->strides[1]);
    dst_p += dst_arr->strides[0];
  }
}

// noise used for dithering (the same for each tile)
static const int dithering_noise_size = 64*64*2;
static uint16_t dithering_noise[dithering_noise_size];
static void precalculate_dithering_noise_if_required()
{
  static bool have_noise = false;
  if (!have_noise) {
    // let's make some noise
    for (int i=0; i<dithering_noise_size; i++) {
      // random number in range [0.03 .. 0.97] * (1<<15)
      //
      // We could use the full range, but like this it is much easier
      // to guarantee 8bpc load-save roundtrips don't alter the
      // image. With the full range we would have to pay a lot
      // attention to rounding converting 8bpc to our internal format.
      dithering_noise[i] = (rand() % (1<<15)) * 240/256 + (1<<15) * 8/256;
    }
    have_noise = true;
  }
}

// used mainly for saving layers (transparent PNG)
void tile_convert_rgba16_to_rgba8(PyObject * src, PyObject * dst) {
  PyArrayObject* src_arr = ((PyArrayObject*)src);
  PyArrayObject* dst_arr = ((PyArrayObject*)dst);

#ifdef HEAVY_DEBUG
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
#endif

  precalculate_dithering_noise_if_required();
  int noise_idx = 0;

  for (int y=0; y<TILE_SIZE; y++) {
    uint16_t * src_p = (uint16_t*)(src_arr->data + y*src_arr->strides[0]);
    uint8_t  * dst_p = (uint8_t*)(dst_arr->data + y*dst_arr->strides[0]);
    for (int x=0; x<TILE_SIZE; x++) {
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
        r = ((r << 15) + a/2) / a;
        g = ((g << 15) + a/2) / a;
        b = ((b << 15) + a/2) / a;
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
      const uint32_t add_r = dithering_noise[noise_idx++];
      const uint32_t add_g = add_r; // hm... do not produce too much color noise
      const uint32_t add_b = add_r;
      const uint32_t add_a = dithering_noise[noise_idx++];

#ifdef HEAVY_DEBUG
      assert(add_a < (1<<15));
      assert(add_a >= 0);
      assert(noise_idx <= dithering_noise_size);
#endif

      *dst_p++ = (r * 255 + add_r) / (1<<15);
      *dst_p++ = (g * 255 + add_g) / (1<<15);
      *dst_p++ = (b * 255 + add_b) / (1<<15);
      *dst_p++ = (a * 255 + add_a) / (1<<15);
    }
    src_p += src_arr->strides[0];
    dst_p += dst_arr->strides[0];
  }
}

// used after compositing (when displaying, or when saving solid PNG or JPG)
void tile_convert_rgbu16_to_rgbu8(PyObject * src, PyObject * dst) {
  PyArrayObject* src_arr = ((PyArrayObject*)src);
  PyArrayObject* dst_arr = ((PyArrayObject*)dst);

#ifdef HEAVY_DEBUG
  assert(PyArray_DIM(dst, 0) == TILE_SIZE);
  assert(PyArray_DIM(dst, 1) == TILE_SIZE);
  assert(PyArray_DIM(dst, 2) == 4);
  assert(PyArray_TYPE(dst) == NPY_UINT8);
  assert(PyArray_ISBEHAVED(dst));
  assert(PyArray_STRIDE(dst, 1) == 4*sizeof(uint8_t));
  assert(PyArray_STRIDE(dst, 2) == sizeof(uint8_t));

  assert(PyArray_DIM(src, 0) == TILE_SIZE);
  assert(PyArray_DIM(src, 1) == TILE_SIZE);
  assert(PyArray_DIM(src, 2) == 4);
  assert(PyArray_TYPE(src) == NPY_UINT16);
  assert(PyArray_ISBEHAVED(src));
  assert(PyArray_STRIDE(src, 1) == 4*sizeof(uint16_t));
  assert(PyArray_STRIDE(src, 2) ==   sizeof(uint16_t));
#endif

  precalculate_dithering_noise_if_required();
  int noise_idx = 0;

  for (int y=0; y<TILE_SIZE; y++) {
    uint16_t * src_p = (uint16_t*)(src_arr->data + y*src_arr->strides[0]);
    uint8_t  * dst_p = (uint8_t*)(dst_arr->data + y*dst_arr->strides[0]);
    for (int x=0; x<TILE_SIZE; x++) {
      uint32_t r, g, b;
      r = *src_p++;
      g = *src_p++;
      b = *src_p++;
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
      const uint32_t add = dithering_noise[noise_idx++];
      
      *dst_p++ = (r * 255 + add) / (1<<15);
      *dst_p++ = (g * 255 + add) / (1<<15);
      *dst_p++ = (b * 255 + add) / (1<<15);
      *dst_p++ = 255;
    }
#ifdef HEAVY_DEBUG
    assert(noise_idx <= dithering_noise_size);
#endif
    src_p += src_arr->strides[0];
    dst_p += dst_arr->strides[0];
  }
}

// used mainly for loading layers (transparent PNG)
void tile_convert_rgba8_to_rgba16(PyObject * src, PyObject * dst) {
  PyArrayObject* src_arr = ((PyArrayObject*)src);
  PyArrayObject* dst_arr = ((PyArrayObject*)dst);

#ifdef HEAVY_DEBUG
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
#endif

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

// used in strokemap.py
//
// Calculates a 1-bit bitmap of the stroke shape using two snapshots
// of the layer (the layer before and after the stroke).
//
// If the alpha increases a lot, we want the stroke to appear in
// the strokemap, even if the color did not change. If the alpha
// decreases a lot, we want to ignore the stroke (eraser). If
// the alpha decreases just a little, but the color changes a
// lot (eg. heavy smudging or watercolor brushes) we want the
// stroke still to be pickable.
//
// If the layer alpha was (near) zero, we record the stroke even if it
// is barely visible. This gives a bigger target to point-and-select.
//
void tile_perceptual_change_strokemap(PyObject * a, PyObject * b, PyObject * res) {

  assert(PyArray_TYPE(a) == NPY_UINT16);
  assert(PyArray_TYPE(b) == NPY_UINT16);
  assert(PyArray_TYPE(res) == NPY_UINT8);
  assert(PyArray_ISCARRAY(a));
  assert(PyArray_ISCARRAY(b));
  assert(PyArray_ISCARRAY(res));

  uint16_t * a_p  = (uint16_t*)PyArray_DATA(a);
  uint16_t * b_p  = (uint16_t*)PyArray_DATA(b);
  uint8_t * res_p = (uint8_t*)PyArray_DATA(res);

  for (int y=0; y<TILE_SIZE; y++) {
    for (int x=0; x<TILE_SIZE; x++) {

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

