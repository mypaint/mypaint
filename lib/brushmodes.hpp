/* This file is part of MyPaint.
 * Copyright (C) 2008-2011 by Martin Renold <martinxyz@gmx.ch>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */


// parameters to those methods:
//
// rgba: A pointer to 16bit rgba data with premultiplied alpha.
//       The range of each components is limited from 0 to 2^15.
//
// mask: Contains the dab shape, that is, the intensity of the dab at
//       each pixel. Usually rendering is done for one tile at a
//       time. The mask is LRE encoded to jump quickly over regions
//       that are not affected by the dab.
//
// opacity: overall strenght of the blending mode. Has the same
//          influence on the dab as the values inside the mask.


// We are manipulating pixels with premultiplied alpha directly.
// This is an "over" operation (opa = topAlpha).
// In the formula below, topColor is assumed to be premultiplied.
//
//               opa_a      <   opa_b      >
// resultAlpha = topAlpha + (1.0 - topAlpha) * bottomAlpha
// resultColor = topColor + (1.0 - topAlpha) * bottomColor
//
void draw_dab_pixels_BlendMode_Normal (uint16_t * mask,
                                       uint16_t * rgba,
                                       uint16_t color_r,
                                       uint16_t color_g,
                                       uint16_t color_b,
                                       uint16_t opacity) {

  while (1) {
    for (; mask[0]; mask++, rgba+=4) {
      uint32_t opa_a = mask[0]*(uint32_t)opacity/(1<<15); // topAlpha
      uint32_t opa_b = (1<<15)-opa_a; // bottomAlpha
      rgba[3] = opa_a + opa_b * rgba[3] / (1<<15);
      rgba[0] = (opa_a*color_r + opa_b*rgba[0])/(1<<15);
      rgba[1] = (opa_a*color_g + opa_b*rgba[1])/(1<<15);
      rgba[2] = (opa_a*color_b + opa_b*rgba[2])/(1<<15);

    }
    if (!mask[1]) break;
    rgba += mask[1];
    mask += 2;
  }
};



// Colorize: apply the source hue and saturation, retaining the target
// brightness. Same thing as in the PDF spec addendum but with different Luma
// coefficients. Colorize should be used at either 1.0 or 0.0, values in
// between probably aren't very useful. This blend mode retains the target
// alpha, and any pure whites and blacks in the target layer.
//
// Code is still ugly. Using floating-point arithmetic in here, which maybe
// isn't very efficient.

#define MAX3(a, b, c) ((a)>(b)?MAX((a),(c)):MAX((b),(c)))
#define MIN3(a, b, c) ((a)<(b)?MIN((a),(c)):MIN((b),(c)))

/*
// From ITU Rec. BT.601 (SDTV)
static const float LUMA_RED_COEFF   = 0.3;
static const float LUMA_GREEN_COEFF = 0.59;
static const float LUMA_BLUE_COEFF  = 0.11;
*/

// From ITU Rec. BT.709 (HDTV and sRGB)
static const float LUMA_RED_COEFF = 0.2126;
static const float LUMA_GREEN_COEFF = 0.7152;
static const float LUMA_BLUE_COEFF = 0.0722;

// See also http://en.wikipedia.org/wiki/YCbCr


inline static float
nonsep_lum (const float r,
              const float g,
              const float b)
{
    return (r*LUMA_RED_COEFF) + (g*LUMA_GREEN_COEFF) + (b*LUMA_BLUE_COEFF);
}


inline static void
nonsep_clip_inplace(float *r,
                    float *g,
                    float *b)
{
    float lum = nonsep_lum(*r, *g, *b);
    float cmin = MIN3(*r, *g, *b);
    float cmax = MAX3(*r, *g, *b);
    if (cmin < 0.0) {
        *r = lum + (((*r - lum) * lum) / (lum - cmin));
        *g = lum + (((*g - lum) * lum) / (lum - cmin));
        *b = lum + (((*b - lum) * lum) / (lum - cmin));
    }
    if (cmax > 1.0) {
        *r = lum + (((*r - lum) * (1-lum)) / (cmax - lum));
        *g = lum + (((*g - lum) * (1-lum)) / (cmax - lum));
        *b = lum + (((*b - lum) * (1-lum)) / (cmax - lum));
    }
}


inline static void
nonsep_apply_lum(const float topr,
                 const float topg,
                 const float topb,
                 const float botlum,
                 float *botr,
                 float *botg,
                 float *botb)
{
    float diff = botlum - nonsep_lum(topr, topg, topb);
    *botr = topr + diff;
    *botg = topg + diff;
    *botb = topb + diff;
    nonsep_clip_inplace(botr, botg, botb);
}


// The method is an implementation of that described in the official Adobe "PDF
// Blend Modes: Addendum" document, dated January 23, 2006; specifically it's
// the "Color" nonseparable blend mode. We do however use different
// coefficients for the Luma value.

void
draw_dab_pixels_BlendMode_Color (uint16_t * mask,
                                 uint16_t * rgba, // b/bottom, premult
                                 uint16_t color_r,  // }
                                 uint16_t color_g,  // }-- a/top, !premult
                                 uint16_t color_b,  // }
                                 uint16_t opacity)
{
  while (1) {
    for (; mask[0]; mask++, rgba+=4) {
      // De-multiply (and scale)
      const float scalefact = rgba[3]; //can't work with premult alpha, sadly

      float r = ((float)rgba[0]) / scalefact;
      float g = ((float)rgba[1]) / scalefact;
      float b = ((float)rgba[2]) / scalefact;

      // Input luma, based on the target pixel
      float lum = nonsep_lum(r, g, b);

      // Output RGB generation
      r = g = b = 0;
      nonsep_apply_lum( (float)color_r/(1<<15),
                        (float)color_g/(1<<15),
                        (float)color_b/(1<<15),
                        lum, &r, &g, &b         );

      // Re-premult/scale.
      r *= scalefact;
      g *= scalefact;
      b *= scalefact;

      // And combine as normal.
      uint32_t opa_a = mask[0]*(uint32_t)opacity/(1<<15); // topAlpha
      uint32_t opa_b = (1<<15)-opa_a; // bottomAlpha
      rgba[0] = (opa_a*r + opa_b*rgba[0])/(1<<15);
      rgba[1] = (opa_a*g + opa_b*rgba[1])/(1<<15);
      rgba[2] = (opa_a*b + opa_b*rgba[2])/(1<<15);
    }
    if (!mask[1]) break;
    rgba += mask[1];
    mask += 2;
  }
};

// This blend mode is used for smudging and erasing.  Smudging
// allows to "drag" around transparency as if it was a color.  When
// smuding over a region that is 60% opaque the result will stay 60%
// opaque (color_a=0.6).  For normal erasing color_a is set to 0.0
// and color_r/g/b will be ignored. This function can also do normal
// blending (color_a=1.0).
//
void draw_dab_pixels_BlendMode_Normal_and_Eraser (uint16_t * mask,
                                                  uint16_t * rgba,
                                                  uint16_t color_r,
                                                  uint16_t color_g,
                                                  uint16_t color_b,
                                                  uint16_t color_a,
                                                  uint16_t opacity) {

  while (1) {
    for (; mask[0]; mask++, rgba+=4) {
      uint32_t opa_a = mask[0]*(uint32_t)opacity/(1<<15); // topAlpha
      uint32_t opa_b = (1<<15)-opa_a; // bottomAlpha
      opa_a = opa_a * color_a / (1<<15);
      rgba[3] = opa_a + opa_b * rgba[3] / (1<<15);
      rgba[0] = (opa_a*color_r + opa_b*rgba[0])/(1<<15);
      rgba[1] = (opa_a*color_g + opa_b*rgba[1])/(1<<15);
      rgba[2] = (opa_a*color_b + opa_b*rgba[2])/(1<<15);

    }
    if (!mask[1]) break;
    rgba += mask[1];
    mask += 2;
  }
};

// This is BlendMode_Normal with locked alpha channel.
//
void draw_dab_pixels_BlendMode_LockAlpha (uint16_t * mask,
                                          uint16_t * rgba,
                                          uint16_t color_r,
                                          uint16_t color_g,
                                          uint16_t color_b,
                                          uint16_t opacity) {

  while (1) {
    for (; mask[0]; mask++, rgba+=4) {
      uint32_t opa_a = mask[0]*(uint32_t)opacity/(1<<15); // topAlpha
      uint32_t opa_b = (1<<15)-opa_a; // bottomAlpha
      
      opa_a *= rgba[3];
      opa_a /= (1<<15);
          
      rgba[0] = (opa_a*color_r + opa_b*rgba[0])/(1<<15);
      rgba[1] = (opa_a*color_g + opa_b*rgba[1])/(1<<15);
      rgba[2] = (opa_a*color_b + opa_b*rgba[2])/(1<<15);
    }
    if (!mask[1]) break;
    rgba += mask[1];
    mask += 2;
  }
};


// Sum up the color/alpha components inside the masked region.
// Called by get_color().
//
void get_color_pixels_accumulate (uint16_t * mask,
                                  uint16_t * rgba,
                                  float * sum_weight,
                                  float * sum_r,
                                  float * sum_g,
                                  float * sum_b,
                                  float * sum_a
                                  ) {


  // The sum of a 64x64 tile fits into a 32 bit integer, but the sum
  // of an arbitrary number of tiles may not fit. We assume that we
  // are processing a single tile at a time, so we can use integers.
  // But for the result we need floats.

  uint32_t weight = 0;
  uint32_t r = 0;
  uint32_t g = 0;
  uint32_t b = 0;
  uint32_t a = 0;

  while (1) {
    for (; mask[0]; mask++, rgba+=4) {
      uint32_t opa = mask[0];
      weight += opa;
      r      += opa*rgba[0]/(1<<15);
      g      += opa*rgba[1]/(1<<15);
      b      += opa*rgba[2]/(1<<15);
      a      += opa*rgba[3]/(1<<15);

    }
    if (!mask[1]) break;
    rgba += mask[1];
    mask += 2;
  }

  // convert integer to float outside the performance critical loop
  *sum_weight += weight;
  *sum_r += r;
  *sum_g += g;
  *sum_b += b;
  *sum_a += a;
};

