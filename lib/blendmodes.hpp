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

#ifndef __HAVE_BLEND_MODES
#define __HAVE_BLEND_MODES

#include "fix15.hpp"
#include "compositing.hpp"


// Normal: http://www.w3.org/TR/compositing/#blendingnormal

class NormalBlendMode
{
  public:
    inline void operator()
        (const fix15_t src_r, const fix15_t src_g, const fix15_t src_b,
         fix15_t &dst_r, fix15_t &dst_g, fix15_t &dst_b)
    {
        dst_r = src_r;
        dst_g = src_g;
        dst_b = src_b;
    }
};


template <unsigned int BUFSIZE>
class BufferComp <BufferCompOutputRGBX, BUFSIZE, NormalBlendMode>
{
    // Partial specialization for for the most common case, working in
    // premultiplied alpha for speed.
  public:
    static inline void composite_src_over
            (const fix15_short_t * const src,
             fix15_short_t * const dst,
             const fix15_short_t opac)
    {
        for (unsigned int i=0; i<BUFSIZE; i+=4) {
            const fix15_t one_minus_Sa = fix15_one - fix15_mul(src[i+3], opac);
            dst[i+0] = fix15_sumprods(src[i], opac, one_minus_Sa, dst[i]);
            dst[i+1] = fix15_sumprods(src[i+1], opac, one_minus_Sa, dst[i+1]);
            dst[i+2] = fix15_sumprods(src[i+2], opac, one_minus_Sa, dst[i+2]);
        }
    }
};



// Multiply: http://www.w3.org/TR/compositing/#blendingmultiply

class MultiplyBlendMode
{
  public:
    inline void operator()
        (const fix15_t src_r, const fix15_t src_g, const fix15_t src_b,
         fix15_t &dst_r, fix15_t &dst_g, fix15_t &dst_b)
    {
        dst_r = fix15_mul(src_r, dst_r);
        dst_g = fix15_mul(src_g, dst_g);
        dst_b = fix15_mul(src_b, dst_b);
    }
};



// Screen: http://www.w3.org/TR/compositing/#blendingscreen

class ScreenBlendMode
{
  public:
    inline void operator()
        (const fix15_t src_r, const fix15_t src_g, const fix15_t src_b,
         fix15_t &dst_r, fix15_t &dst_g, fix15_t &dst_b)
    {
        dst_r = dst_r + src_r - fix15_mul(dst_r, src_r);
        dst_g = dst_g + src_g - fix15_mul(dst_g, src_g);
        dst_b = dst_b + src_b - fix15_mul(dst_b, src_b);
    }
};



// Overlay: http://www.w3.org/TR/compositing/#blendingoverlay

class OverlayBlendMode
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
         fix15_t &dst_r, fix15_t &dst_g, fix15_t &dst_b)
    {
        process_channel(src_r, dst_r);
        process_channel(src_g, dst_g);
        process_channel(src_b, dst_b);
    }
};


// Darken: http://www.w3.org/TR/compositing/#blendingdarken

class DarkenBlendMode
{
  public:
    inline void operator()
        (const fix15_t src_r, const fix15_t src_g, const fix15_t src_b,
         fix15_t &dst_r, fix15_t &dst_g, fix15_t &dst_b)
    {
        if (src_r < dst_r) dst_r = src_r;
        if (src_g < dst_g) dst_g = src_g;
        if (src_b < dst_b) dst_b = src_b;
    }
};


// Lighten: http://www.w3.org/TR/compositing/#blendinglighten

class LightenBlendMode
{
  public:
    inline void operator()
        (const fix15_t src_r, const fix15_t src_g, const fix15_t src_b,
         fix15_t &dst_r, fix15_t &dst_g, fix15_t &dst_b)
    {
        if (src_r > dst_r) dst_r = src_r;
        if (src_g > dst_g) dst_g = src_g;
        if (src_b > dst_b) dst_b = src_b;
    }
};



// Hard Light: http://www.w3.org/TR/compositing/#blendinghardlight

class HardLightBlendMode
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
         fix15_t &dst_r, fix15_t &dst_g, fix15_t &dst_b)
    {
        process_channel(src_r, dst_r);
        process_channel(src_g, dst_g);
        process_channel(src_b, dst_b);
    }
};


// Color-dodge: http://www.w3.org/TR/compositing/#blendingcolordodge

class ColorDodgeBlendMode
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
         fix15_t &dst_r, fix15_t &dst_g, fix15_t &dst_b)
    {
        process_channel(src_r, dst_r);
        process_channel(src_g, dst_g);
        process_channel(src_b, dst_b);
    }
};


// Color-burn: http://www.w3.org/TR/compositing/#blendingcolorburn

class ColorBurnBlendMode
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
         fix15_t &dst_r, fix15_t &dst_g, fix15_t &dst_b)
    {
        process_channel(src_r, dst_r);
        process_channel(src_g, dst_g);
        process_channel(src_b, dst_b);
    }
};


// Soft-light: http://www.w3.org/TR/compositing/#blendingsoftlight

class SoftLightBlendMode
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
         fix15_t &dst_r, fix15_t &dst_g, fix15_t &dst_b)
    {
        process_channel(src_r, dst_r);
        process_channel(src_g, dst_g);
        process_channel(src_b, dst_b);
    }
};


// Difference: http://www.w3.org/TR/compositing/#blendingdifference

class DifferenceBlendMode
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
         fix15_t &dst_r, fix15_t &dst_g, fix15_t &dst_b)
    {
        process_channel(src_r, dst_r);
        process_channel(src_g, dst_g);
        process_channel(src_b, dst_b);
    }
};


// Exclusion: http://www.w3.org/TR/compositing/#blendingexclusion

class ExclusionBlendMode
{
  private:
    static inline void process_channel(const fix15_t Cs, fix15_t &Cb)
    {
        Cb = Cb + Cs - fix15_double(fix15_mul(Cb, Cs));
    }

  public:
    inline void operator()
        (const fix15_t src_r, const fix15_t src_g, const fix15_t src_b,
         fix15_t &dst_r, fix15_t &dst_g, fix15_t &dst_b)
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

class HueBlendMode
{
  public:
    inline void operator()
        (const fix15_t src_r, const fix15_t src_g, const fix15_t src_b,
         fix15_t &dst_r, fix15_t &dst_g, fix15_t &dst_b)
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

class SaturationBlendMode
{
  public:
    inline void operator()
        (const fix15_t src_r, const fix15_t src_g, const fix15_t src_b,
         fix15_t &dst_r, fix15_t &dst_g, fix15_t &dst_b)
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

class ColorBlendMode
{
  public:
    inline void operator()
        (const fix15_t src_r, const fix15_t src_g, const fix15_t src_b,
         fix15_t &dst_r, fix15_t &dst_g, fix15_t &dst_b)
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

class LuminosityBlendMode
{
  public:
    inline void operator()
        (const fix15_t src_r, const fix15_t src_g, const fix15_t src_b,
         fix15_t &dst_r, fix15_t &dst_g, fix15_t &dst_b)
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


#endif //__HAVE_BLEND_MODES
