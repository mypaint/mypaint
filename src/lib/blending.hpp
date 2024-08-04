/* This file is part of MyPaint.
 * Copyright (C) 2012 by Andrew Chadwick <a.t.chadwick@gmail.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */

// Generic blend mode functors, with partially specialized buffer compositors
// for some optimized cases.

#ifndef __HAVE_BLENDING
#define __HAVE_BLENDING
#define WGM_EPSILON 0.001

#include "fastapprox/fastpow.h"
#include "fix15.hpp"
#include "compositing.hpp"

static const float T_MATRIX_SMALL[3][10] = {{0.026595621243689,0.049779426257903,0.022449850859496,-0.218453689278271
,-0.256894883201278,0.445881722194840,0.772365886289756,0.194498761382537
,0.014038157587820,0.007687264480513}
,{-0.032601672674412,-0.061021043498478,-0.052490001018404
,0.206659098273522,0.572496335158169,0.317837248815438,-0.021216624031211
,-0.019387668756117,-0.001521339050858,-0.000835181622534}
,{0.339475473216284,0.635401374177222,0.771520797089589,0.113222640692379
,-0.055251113343776,-0.048222578468680,-0.012966666339586
,-0.001523814504223,-0.000094718948810,-0.000051604594741}};

static const float spectral_r_small[10] = {0.009281362787953,0.009732627042016,0.011254252737167,0.015105578649573
,0.024797924177217,0.083622585502406,0.977865045723212,1.000000000000000
,0.999961046144372,0.999999992756822};

static const float spectral_g_small[10] = {0.002854127435775,0.003917589679914,0.012132151699187,0.748259205918013
,1.000000000000000,0.865695937531795,0.037477469241101,0.022816789725717
,0.021747419446456,0.021384940572308};

static const float spectral_b_small[10] = {0.537052150373386,0.546646402401469,0.575501819073983,0.258778829633924
,0.041709923751716,0.012662638828324,0.007485593127390,0.006766900622462
,0.006699764779016,0.006676219883241};


void
rgb_to_spectral (float r, float g, float b, float *spectral_) {
  float offset = 1.0 - WGM_EPSILON;
  r = r * offset + WGM_EPSILON;
  g = g * offset + WGM_EPSILON;
  b = b * offset + WGM_EPSILON;
  //upsample rgb to spectral primaries
  float spec_r[10] = {0};
  for (int i=0; i < 10; i++) {
    spec_r[i] = spectral_r_small[i] * r;
  }
  float spec_g[10] = {0};
  for (int i=0; i < 10; i++) {
    spec_g[i] = spectral_g_small[i] * g;
  }
  float spec_b[10] = {0};
  for (int i=0; i < 10; i++) {
    spec_b[i] = spectral_b_small[i] * b;
  }
  //collapse into one spd
  for (int i=0; i<10; i++) {
    spectral_[i] += spec_r[i] + spec_g[i] + spec_b[i];
  }

}

void
spectral_to_rgb (float *spectral, float *rgb_) {
  float offset = 1.0 - WGM_EPSILON;
  for (int i=0; i<10; i++) {
    rgb_[0] += T_MATRIX_SMALL[0][i] * spectral[i];
    rgb_[1] += T_MATRIX_SMALL[1][i] * spectral[i];
    rgb_[2] += T_MATRIX_SMALL[2][i] * spectral[i];
  }
  for (int i=0; i<3; i++) {
    rgb_[i] = CLAMP((rgb_[i] - WGM_EPSILON) / offset, 0.0f, 1.0f);
  }
}



// Normal: http://www.w3.org/TR/compositing/#blendingnormal

class BlendNormal : public BlendFunc
{
  public:
    inline void operator()
        (const fix15_t src_r, const fix15_t src_g, const fix15_t src_b,
         fix15_t &dst_r, fix15_t &dst_g, fix15_t &dst_b) const
    {
        dst_r = src_r;
        dst_g = src_g;
        dst_b = src_b;
    }
};

template <bool DSTALPHA, unsigned int BUFSIZE>
class BufferCombineFunc <DSTALPHA, BUFSIZE, BlendNormal, CompositeSourceOver>
{
    // Partial specialization for normal painting layers (svg:src-over),
    // working in premultiplied alpha for speed.
  public:
    inline void operator() (const fix15_short_t * const src,
                            fix15_short_t * const dst,
                            const fix15_short_t opac) const
    {
        for (unsigned int i=0; i<BUFSIZE; i+=4) {
            const fix15_t Sa = fix15_mul(src[i+3], opac);
            const fix15_t one_minus_Sa = fix15_one - Sa;
            dst[i+0] = fix15_sumprods(src[i], opac, one_minus_Sa, dst[i]);
            dst[i+1] = fix15_sumprods(src[i+1], opac, one_minus_Sa, dst[i+1]);
            dst[i+2] = fix15_sumprods(src[i+2], opac, one_minus_Sa, dst[i+2]);
            if (DSTALPHA) {
                dst[i+3] = fix15_short_clamp(Sa + fix15_mul(dst[i+3], one_minus_Sa));
            }
        }
    }
};

template <bool DSTALPHA, unsigned int BUFSIZE>
class BufferCombineFunc <DSTALPHA, BUFSIZE, BlendNormal, CompositeSpectralWGM>
{
    // Spectral Upsampled Weighted Geometric Mean Pigment/Paint Emulation
    // Based on work by Scott Allen Burns, Meng, and others.
  public:
    inline void operator() (const fix15_short_t * const src,
                            fix15_short_t * const dst,
                            const fix15_short_t opac) const
    {
        for (unsigned int i=0; i<BUFSIZE; i+=4) {
            const fix15_t Sa = fix15_mul(src[i+3], opac);
            const fix15_t one_minus_Sa = fix15_one - Sa;
            if ((DSTALPHA && dst[i+3] == 0)|| Sa == (1<<15) || Sa == 0) {
              dst[i+0] = fix15_sumprods(src[i], opac, one_minus_Sa, dst[i]);
              dst[i+1] = fix15_sumprods(src[i+1], opac, one_minus_Sa, dst[i+1]);
              dst[i+2] = fix15_sumprods(src[i+2], opac, one_minus_Sa, dst[i+2]);
              if (DSTALPHA) {
                dst[i+3] = fix15_short_clamp(Sa + fix15_mul(dst[i+3], one_minus_Sa));
              }   
            } else {
              //alpha-weighted ratio for WGM (sums to 1.0)
              //fix15_t dst_alpha = (1<<15);
              float fac_a;
              if (DSTALPHA) {
                fac_a = (float)Sa / (Sa + one_minus_Sa * dst[i+3] / (1<<15));
              } else {
                fac_a = (float)Sa / (1<<15);
              }
              float fac_b = 1.0 - fac_a;

              //convert bottom to spectral.  Un-premult alpha to obtain reflectance
              //color noise is not a problem since low alpha also implies low weight
              float spectral_b[10] = {0};
              if (DSTALPHA && dst[i+3] > 0) {
                rgb_to_spectral((float)dst[i] / dst[i+3], (float)dst[i+1] / dst[i+3], (float)dst[i+2] / dst[i+3], spectral_b);
              } else {
                rgb_to_spectral((float)dst[i]/ (1<<15), (float)dst[i+1]/ (1<<15), (float)dst[i+2]/ (1<<15), spectral_b);
              }
              // convert top to spectral.  Already straight color
              float spectral_a[10] = {0};
              if (src[i+3] > 0) {
                rgb_to_spectral((float)src[i] / src[i+3], (float)src[i+1] / src[i+3], (float)src[i+2] / src[i+3], spectral_a);
              } else {
                rgb_to_spectral((float)src[i] / (1<<15), (float)src[i+1] / (1<<15), (float)src[i+2] / (1<<15), spectral_a);
              }
              // mix to the two spectral reflectances using WGM
              float spectral_result[10] = {0};
              for (int i=0; i<10; i++) {
                spectral_result[i] = fastpow(spectral_a[i], fac_a) * fastpow(spectral_b[i], fac_b);
              }
              
              // convert back to RGB and premultiply alpha
              float rgb_result[4] = {0};
              spectral_to_rgb(spectral_result, rgb_result);
              if (DSTALPHA) {
                rgb_result[3] = fix15_short_clamp(Sa + fix15_mul(dst[i+3], one_minus_Sa));
              } else {
                rgb_result[3] = (1<<15);
              }
              for (int j=0; j<3; j++) {
                dst[i+j] =(rgb_result[j] * (rgb_result[3] + 0.5));
              }
              
              if (DSTALPHA) {
                  dst[i+3] = rgb_result[3];
              }            
            }

        }
    }
};

template <bool DSTALPHA, unsigned int BUFSIZE>
class BufferCombineFunc <DSTALPHA, BUFSIZE, BlendNormal, CompositeDestinationIn>
{
    // Partial specialization for svg:dst-in layers,
    // working in premultiplied alpha for speed.
  public:
    inline void operator() (const fix15_short_t * const src,
                            fix15_short_t * const dst,
                            const fix15_short_t opac) const
    {
        for (unsigned int i=0; i<BUFSIZE; i+=4) {
            const fix15_t Sa = fix15_mul(src[i+3], opac);
            dst[i+0] = fix15_mul(dst[i+0], Sa);
            dst[i+1] = fix15_mul(dst[i+1], Sa);
            dst[i+2] = fix15_mul(dst[i+2], Sa);
            if (DSTALPHA) {
                dst[i+3] = fix15_mul(Sa, dst[i+3]);
            }
        }
    }
};

template <bool DSTALPHA, unsigned int BUFSIZE>
class BufferCombineFunc <DSTALPHA, BUFSIZE, BlendNormal, CompositeDestinationOut>
{
    // Partial specialization for svg:dst-out layers,
    // working in premultiplied alpha for speed.
  public:
    inline void operator() (const fix15_short_t * const src,
                            fix15_short_t * const dst,
                            const fix15_short_t opac) const
    {
        for (unsigned int i=0; i<BUFSIZE; i+=4) {
            const fix15_t one_minus_Sa = fix15_one-fix15_mul(src[i+3], opac);
            dst[i+0] = fix15_mul(dst[i+0], one_minus_Sa);
            dst[i+1] = fix15_mul(dst[i+1], one_minus_Sa);
            dst[i+2] = fix15_mul(dst[i+2], one_minus_Sa);
            if (DSTALPHA) {
                dst[i+3] = fix15_mul(one_minus_Sa, dst[i+3]);
            }
        }
    }
};


template <bool DSTALPHA, unsigned int BUFSIZE>
class BufferCombineFunc <DSTALPHA, BUFSIZE, BlendNormal, CompositeSourceAtop>
{
    // Partial specialization for svg:src-atop layers,
    // working in premultiplied alpha for speed.
  public:
    inline void operator() (const fix15_short_t * const src,
                            fix15_short_t * const dst,
                            const fix15_short_t opac) const
    {
        for (unsigned int i=0; i<BUFSIZE; i+=4) {
            const fix15_t as = fix15_mul(src[i+3], opac);
            const fix15_t ab = dst[i+3];
            const fix15_t one_minus_as = fix15_one - as;
            // W3C spec:
            //   co = as*Cs*ab + ab*Cb*(1-as)
            // where
            //   src[n] = as*Cs    -- premultiplied
            //   dst[n] = ab*Cb    -- premultiplied
            dst[i+0] = fix15_sumprods(fix15_mul(src[i+0], opac), ab,
                                      dst[i+0], one_minus_as);
            dst[i+1] = fix15_sumprods(fix15_mul(src[i+1], opac), ab,
                                      dst[i+1], one_minus_as);
            dst[i+2] = fix15_sumprods(fix15_mul(src[i+2], opac), ab,
                                      dst[i+2], one_minus_as);
            // W3C spec:
            //   ao = as*ab + ab*(1-as)
            //   ao = ab
            // (leave output alpha unchanged)
        }
    }
};

template <bool DSTALPHA, unsigned int BUFSIZE>
class BufferCombineFunc <DSTALPHA, BUFSIZE, BlendNormal, CompositeDestinationAtop>
{
    // Partial specialization for svg:dst-atop layers,
    // working in premultiplied alpha for speed.
  public:
    inline void operator() (const fix15_short_t * const src,
                            fix15_short_t * const dst,
                            const fix15_short_t opac) const
    {
        for (unsigned int i=0; i<BUFSIZE; i+=4) {
            const fix15_t as = fix15_mul(src[i+3], opac);
            const fix15_t ab = dst[i+3];
            const fix15_t one_minus_ab = fix15_one - ab;
            // W3C Spec:
            //   co = as*Cs*(1-ab) + ab*Cb*as
            // where
            //   src[n] = as*Cs    -- premultiplied
            //   dst[n] = ab*Cb    -- premultiplied
            dst[i+0] = fix15_sumprods(fix15_mul(src[i+0], opac), one_minus_ab,
                                      dst[i+0], as);
            dst[i+1] = fix15_sumprods(fix15_mul(src[i+1], opac), one_minus_ab,
                                      dst[i+1], as);
            dst[i+2] = fix15_sumprods(fix15_mul(src[i+2], opac), one_minus_ab,
                                      dst[i+2], as);
            // W3C spec:
            //   ao = as*(1-ab) + ab*as
            //   ao = as
            if (DSTALPHA) {
                dst[i+3] = as;
            }
        }
    }
};


// Multiply: http://www.w3.org/TR/compositing/#blendingmultiply

class BlendMultiply : public BlendFunc
{
  public:
    inline void operator()
        (const fix15_t src_r, const fix15_t src_g, const fix15_t src_b,
         fix15_t &dst_r, fix15_t &dst_g, fix15_t &dst_b) const
    {
        dst_r = fix15_mul(src_r, dst_r);
        dst_g = fix15_mul(src_g, dst_g);
        dst_b = fix15_mul(src_b, dst_b);
    }
};




// Screen: http://www.w3.org/TR/compositing/#blendingscreen

class BlendScreen : public BlendFunc
{
  public:
    inline void operator()
        (const fix15_t src_r, const fix15_t src_g, const fix15_t src_b,
         fix15_t &dst_r, fix15_t &dst_g, fix15_t &dst_b) const
    {
        dst_r = dst_r + src_r - fix15_mul(dst_r, src_r);
        dst_g = dst_g + src_g - fix15_mul(dst_g, src_g);
        dst_b = dst_b + src_b - fix15_mul(dst_b, src_b);
    }
};



// Overlay: http://www.w3.org/TR/compositing/#blendingoverlay

class BlendOverlay : public BlendFunc
{
  private:
    static inline void process_channel(const fix15_t Cs, fix15_t &Cb)
    {
        const fix15_t two_Cb = fix15_double(Cb);
        if (two_Cb <= fix15_one) {
            Cb = fix15_mul(Cs, two_Cb);
        }
        else {
            const fix15_t tmp = two_Cb - fix15_one;
            Cb = Cs + tmp - fix15_mul(Cs, tmp);
        }
    }

  public:
    inline void operator()
        (const fix15_t src_r, const fix15_t src_g, const fix15_t src_b,
         fix15_t &dst_r, fix15_t &dst_g, fix15_t &dst_b) const
    {
        process_channel(src_r, dst_r);
        process_channel(src_g, dst_g);
        process_channel(src_b, dst_b);
    }
};


// Darken: http://www.w3.org/TR/compositing/#blendingdarken

class BlendDarken : public BlendFunc
{
  public:
    inline void operator()
        (const fix15_t src_r, const fix15_t src_g, const fix15_t src_b,
         fix15_t &dst_r, fix15_t &dst_g, fix15_t &dst_b) const
    {
        if (src_r < dst_r) dst_r = src_r;
        if (src_g < dst_g) dst_g = src_g;
        if (src_b < dst_b) dst_b = src_b;
    }
};


// Lighten: http://www.w3.org/TR/compositing/#blendinglighten

class BlendLighten : public BlendFunc
{
  public:
    inline void operator()
        (const fix15_t src_r, const fix15_t src_g, const fix15_t src_b,
         fix15_t &dst_r, fix15_t &dst_g, fix15_t &dst_b) const
    {
        if (src_r > dst_r) dst_r = src_r;
        if (src_g > dst_g) dst_g = src_g;
        if (src_b > dst_b) dst_b = src_b;
    }
};



// Hard Light: http://www.w3.org/TR/compositing/#blendinghardlight

class BlendHardLight : public BlendFunc
{
  private:
    static inline void process_channel(const fix15_t Cs, fix15_t &Cb)
    {
        const fix15_t two_Cs = fix15_double(Cs);
        if (two_Cs <= fix15_one) {
            Cb = fix15_mul(Cb, two_Cs);
        }
        else {
            const fix15_t tmp = two_Cs - fix15_one;
            Cb = Cb + tmp - fix15_mul(Cb, tmp);
        }
    }

  public:
    inline void operator()
        (const fix15_t src_r, const fix15_t src_g, const fix15_t src_b,
         fix15_t &dst_r, fix15_t &dst_g, fix15_t &dst_b) const
    {
        process_channel(src_r, dst_r);
        process_channel(src_g, dst_g);
        process_channel(src_b, dst_b);
    }
};


// Color-dodge: http://www.w3.org/TR/compositing/#blendingcolordodge

class BlendColorDodge : public BlendFunc
{
  private:
    static inline void process_channel(const fix15_t Cs, fix15_t &Cb)
    {
        if (Cs < fix15_one) {
            const fix15_t tmp = fix15_div(Cb, fix15_one - Cs);
            if (tmp < fix15_one) {
                Cb = tmp;
                return;
            }
        }
        Cb = fix15_one;
    }

  public:
    inline void operator()
        (const fix15_t src_r, const fix15_t src_g, const fix15_t src_b,
         fix15_t &dst_r, fix15_t &dst_g, fix15_t &dst_b) const
    {
        process_channel(src_r, dst_r);
        process_channel(src_g, dst_g);
        process_channel(src_b, dst_b);
    }
};


// Color-burn: http://www.w3.org/TR/compositing/#blendingcolorburn

class BlendColorBurn : public BlendFunc
{
  private:
    static inline void process_channel(const fix15_t Cs, fix15_t &Cb)
    {
        if (Cs > 0) {
            const fix15_t tmp = fix15_div(fix15_one - Cb, Cs);
            if (tmp < fix15_one) {
                Cb = fix15_one - tmp;
                return;
            }
        }
        Cb = 0;
    }

  public:
    inline void operator()
        (const fix15_t src_r, const fix15_t src_g, const fix15_t src_b,
         fix15_t &dst_r, fix15_t &dst_g, fix15_t &dst_b) const
    {
        process_channel(src_r, dst_r);
        process_channel(src_g, dst_g);
        process_channel(src_b, dst_b);
    }
};


// Soft-light: http://www.w3.org/TR/compositing/#blendingsoftlight

class BlendSoftLight : public BlendFunc
{
  private:
    static inline void process_channel(const fix15_t Cs, fix15_t &Cb)
    {
        const fix15_t two_Cs = fix15_double(Cs);
        fix15_t B = 0;
        if (two_Cs <= fix15_one) {
            B = fix15_one - fix15_mul(fix15_one - two_Cs,
                                      fix15_one - Cb);
            B = fix15_mul(B, Cb);
        }
        else {
            fix15_t D = 0;
            const fix15_t four_Cb = Cb << 2;
            if (four_Cb <= fix15_one) {
                const fix15_t Cb_squared = fix15_mul(Cb, Cb);
                D = four_Cb; /* which is always greater than... */
                D += 16 * fix15_mul(Cb_squared, Cb);
                D -= 12 * Cb_squared;
                /* ... in the range 0 <= C_b <= 0.25 */
            }
            else {
                D = fix15_sqrt(Cb);
            }
#ifdef HEAVY_DEBUG
            /* Guard against underflows */
            assert(two_Cs > fix15_one);
            assert(D >= Cb);
#endif
            B = Cb + fix15_mul(2*Cs - fix15_one /* 2*Cs > 1 */,
                               D - Cb           /* D >= Cb */  );
        }
        Cb = B;
    }

  public:
    inline void operator()
        (const fix15_t src_r, const fix15_t src_g, const fix15_t src_b,
         fix15_t &dst_r, fix15_t &dst_g, fix15_t &dst_b) const
    {
        process_channel(src_r, dst_r);
        process_channel(src_g, dst_g);
        process_channel(src_b, dst_b);
    }
};


// Difference: http://www.w3.org/TR/compositing/#blendingdifference

class BlendDifference : public BlendFunc
{
  private:
    static inline void process_channel(const fix15_t Cs, fix15_t &Cb)
    {
        if (Cs >= Cb)
            Cb = Cs - Cb;
        else
            Cb = Cb - Cs;
    }

  public:
    inline void operator()
        (const fix15_t src_r, const fix15_t src_g, const fix15_t src_b,
         fix15_t &dst_r, fix15_t &dst_g, fix15_t &dst_b) const
    {
        process_channel(src_r, dst_r);
        process_channel(src_g, dst_g);
        process_channel(src_b, dst_b);
    }
};


// Exclusion: http://www.w3.org/TR/compositing/#blendingexclusion

class BlendExclusion : public BlendFunc
{
  private:
    static inline void process_channel(const fix15_t Cs, fix15_t &Cb)
    {
        Cb = Cb + Cs - fix15_double(fix15_mul(Cb, Cs));
    }

  public:
    inline void operator()
        (const fix15_t src_r, const fix15_t src_g, const fix15_t src_b,
         fix15_t &dst_r, fix15_t &dst_g, fix15_t &dst_b) const
    {
        process_channel(src_r, dst_r);
        process_channel(src_g, dst_g);
        process_channel(src_b, dst_b);
    }
};



//
// Non-separable modes
// http://www.w3.org/TR/compositing/#blendingnonseparable
//

// Auxiliary functions

typedef int32_t ufix15_t;

static const uint16_t BLENDING_LUM_R_COEFF = 0.3  * fix15_one;
static const uint16_t BLENDING_LUM_G_COEFF = 0.59 * fix15_one;
static const uint16_t BLENDING_LUM_B_COEFF = 0.11 * fix15_one;


static inline const ufix15_t
blending_nonsep_lum (const ufix15_t r,
                     const ufix15_t g,
                     const ufix15_t b)
{
    return (  (r) * BLENDING_LUM_R_COEFF
            + (g) * BLENDING_LUM_G_COEFF
            + (b) * BLENDING_LUM_B_COEFF) / fix15_one;
}


static inline void
blending_nonsel_clipcolor (ufix15_t &r,
                           ufix15_t &g,
                           ufix15_t &b)
{
    const ufix15_t lum = blending_nonsep_lum(r, g, b);
    const ufix15_t cmin = (r < g) ? MIN(r, b) : MIN(g, b);
    const ufix15_t cmax = (r > g) ? MAX(r, b) : MAX(g, b);
    if (cmin < 0) {
        const int32_t lum_minus_cmin = lum - cmin;
        r = lum + (((r - lum) * lum) / lum_minus_cmin);
        g = lum + (((g - lum) * lum) / lum_minus_cmin);
        b = lum + (((b - lum) * lum) / lum_minus_cmin);
    }
    if (cmax > (int32_t)fix15_one) {
        const int32_t one_minus_lum = fix15_one - lum;
        const int32_t cmax_minus_lum = cmax - lum;
        r = lum + (((r - lum) * one_minus_lum) / cmax_minus_lum);
        g = lum + (((g - lum) * one_minus_lum) / cmax_minus_lum);
        b = lum + (((b - lum) * one_minus_lum) / cmax_minus_lum);
    }
}


static inline void
blending_nonsep_setlum (ufix15_t &r,
                        ufix15_t &g,
                        ufix15_t &b,
                        const ufix15_t lum)
{
    const ufix15_t diff = lum - blending_nonsep_lum(r, g, b);
    r += diff;
    g += diff;
    b += diff;
    blending_nonsel_clipcolor(r, g, b);
}


static inline const ufix15_t
blending_nonsep_sat (const ufix15_t r,
                     const ufix15_t g,
                     const ufix15_t b)
{
    const ufix15_t cmax = (r > g) ? MAX(r, b) : MAX(g, b);
    const ufix15_t cmin = (r < g) ? MIN(r, b) : MIN(g, b);
    return cmax - cmin;
}


static inline void
blending_nonsep_setsat (ufix15_t &r,
                        ufix15_t &g,
                        ufix15_t &b,
                        const ufix15_t s)
{
    ufix15_t *top_c = &b;
    ufix15_t *mid_c = &g;
    ufix15_t *bot_c = &r;
    ufix15_t *tmp = NULL;
    if (*top_c < *mid_c) { tmp = top_c; top_c = mid_c; mid_c = tmp; }
    if (*top_c < *bot_c) { tmp = top_c; top_c = bot_c; bot_c = tmp; }
    if (*mid_c < *bot_c) { tmp = mid_c; mid_c = bot_c; bot_c = tmp; }
#ifdef HEAVY_DEBUG
    assert(top_c != mid_c);
    assert(mid_c != bot_c);
    assert(bot_c != top_c);
    assert(*top_c >= *mid_c);
    assert(*mid_c >= *bot_c);
    assert(*top_c >= *bot_c);
#endif
    if (*top_c > *bot_c) {
        *mid_c = (*mid_c - *bot_c) * s;  // up to fix30
        *mid_c /= *top_c - *bot_c;       // back down to fix15
        *top_c = s;
    }
    else {
        *top_c = *mid_c = 0;
    }
    *bot_c = 0;
}


// Hue: http://www.w3.org/TR/compositing/#blendinghue

class BlendHue : public BlendFunc
{
  public:
    inline void operator()
        (const fix15_t src_r, const fix15_t src_g, const fix15_t src_b,
         fix15_t &dst_r, fix15_t &dst_g, fix15_t &dst_b) const
    {
        const ufix15_t dst_lum = blending_nonsep_lum(dst_r, dst_g, dst_b);
        const ufix15_t dst_sat = blending_nonsep_sat(dst_r, dst_g, dst_b);
        ufix15_t r = src_r;
        ufix15_t g = src_g;
        ufix15_t b = src_b;
        blending_nonsep_setsat(r, g, b, dst_sat);
        blending_nonsep_setlum(r, g, b, dst_lum);
#ifdef HEAVY_DEBUG
        assert(r <= (ufix15_t)fix15_one);
        assert(g <= (ufix15_t)fix15_one);
        assert(b <= (ufix15_t)fix15_one);
        assert(r >= 0);
        assert(g >= 0);
        assert(b >= 0);
#endif
        dst_r = r;
        dst_g = g;
        dst_b = b;
    }
};


// Saturation: http://www.w3.org/TR/compositing/#blendingsaturation

class BlendSaturation : public BlendFunc
{
  public:
    inline void operator()
        (const fix15_t src_r, const fix15_t src_g, const fix15_t src_b,
         fix15_t &dst_r, fix15_t &dst_g, fix15_t &dst_b) const
    {
        const ufix15_t dst_lum = blending_nonsep_lum(dst_r, dst_g, dst_b);
        const ufix15_t src_sat = blending_nonsep_sat(src_r, src_g, src_b);
        ufix15_t r = dst_r;
        ufix15_t g = dst_g;
        ufix15_t b = dst_b;
        blending_nonsep_setsat(r, g, b, src_sat);
        blending_nonsep_setlum(r, g, b, dst_lum);
#ifdef HEAVY_DEBUG
        assert(r <= (ufix15_t)fix15_one);
        assert(g <= (ufix15_t)fix15_one);
        assert(b <= (ufix15_t)fix15_one);
        assert(r >= 0);
        assert(g >= 0);
        assert(b >= 0);
#endif
        dst_r = r;
        dst_g = g;
        dst_b = b;
    }
};


// Color: http://www.w3.org/TR/compositing/#blendingcolor

class BlendColor : public BlendFunc
{
  public:
    inline void operator()
        (const fix15_t src_r, const fix15_t src_g, const fix15_t src_b,
         fix15_t &dst_r, fix15_t &dst_g, fix15_t &dst_b) const
    {
        ufix15_t r = src_r;
        ufix15_t g = src_g;
        ufix15_t b = src_b;
        blending_nonsep_setlum(r, g, b,
          blending_nonsep_lum(dst_r, dst_g, dst_b));
#ifdef HEAVY_DEBUG
        assert(r <= (ufix15_t)fix15_one);
        assert(g <= (ufix15_t)fix15_one);
        assert(b <= (ufix15_t)fix15_one);
        assert(r >= 0);
        assert(g >= 0);
        assert(b >= 0);
#endif
        dst_r = r;
        dst_g = g;
        dst_b = b;
    }
};


// Luminosity http://www.w3.org/TR/compositing/#blendingluminosity

class BlendLuminosity : public BlendFunc
{
  public:
    inline void operator()
        (const fix15_t src_r, const fix15_t src_g, const fix15_t src_b,
         fix15_t &dst_r, fix15_t &dst_g, fix15_t &dst_b) const
    {
        ufix15_t r = dst_r;
        ufix15_t g = dst_g;
        ufix15_t b = dst_b;
        blending_nonsep_setlum(r, g, b,
          blending_nonsep_lum(src_r, src_g, src_b));
#ifdef HEAVY_DEBUG
        assert(r <= (ufix15_t)fix15_one);
        assert(g <= (ufix15_t)fix15_one);
        assert(b <= (ufix15_t)fix15_one);
        assert(r >= 0);
        assert(g >= 0);
        assert(b >= 0);
#endif
        dst_r = r;
        dst_g = g;
        dst_b = b;
    }
};



#endif //__HAVE_BLENDING
