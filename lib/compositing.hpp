/* This file is part of MyPaint.
 * Copyright (C) 2014 by Andrew Chadwick <a.t.chadwick@gmail.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */

// Template functions for compositing buffers

#ifndef __HAVE_COMPOSITING
#define __HAVE_COMPOSITING

#include "fix15.hpp"

#include <glib.h>


// Abstract interface for TileDataCombine<> blend mode functors
//
// Blend functors are low-level pixel operations. Derived classes' operator()
// implementations should be declared inline and in the class body.
//
// These functors that apply a source colour to a destination, with no
// metadata. The R, G and B values are not premultiplied by alpha during the
// blending phase.

class BlendFunc
{
  public:
    virtual void operator() ( const fix15_t src_r,
                              const fix15_t src_g,
                              const fix15_t src_b,
                              fix15_t &dst_r,
                              fix15_t &dst_g,
                              fix15_t &dst_b ) const = 0;
};


// Abstract interface for TileDataCombine<> compositing op functors
//
// Compositing functors are low-level pixel operations. Derived classes'
// operator() implementations should be declared inline and in the class body.
//
// These are primarily stateless functors which apply a source colour and pixel
// alpha to a destination. At this phase in the rendering workflow, the input
// R, G, and B values are not muliplied by their corresponding A, but the
// output pixel's R, G and B values are multiplied by alpha, and must be
// written as such.
//
// Implementations must also supply details which allow C++ pixel-level
// operations and Python tile-level operations to optimize away blank data or
// skip the dst_has_alpha speedup when necessary.

class CompositeFunc
{
  public:
    virtual void operator() (const fix15_t Rs, const fix15_t Gs,
                             const fix15_t Bs, const fix15_t as,
                             fix15_short_t &rb, fix15_short_t &gb,
                             fix15_short_t &bb, fix15_short_t &ab) const = 0;
    static const bool zero_alpha_has_effect = true;
    static const bool can_decrease_alpha = true;
    static const bool zero_alpha_clears_backdrop = true;
};


// Composable blend+composite functor for buffers
//
// The template parameters define whether the destination's alpha is used,
// and supply the BlendFunc and CompositeFunc functor classes to use.  The
// size of the buffers to be processed must also be specified.
//
// This is templated at the class level so that more optimal partial template
// specializations can be written for more common code paths. The C++ spec
// does not permit plain functions to be partially specialized.
//
// Ref: http://www.w3.org/TR/compositing-1/#generalformula

template <bool DSTALPHA,
          unsigned int BUFSIZE,
          class BLENDFUNC,
          class COMPOSITEFUNC>
class BufferCombineFunc
{
  private:
    BLENDFUNC blendfunc;
    COMPOSITEFUNC compositefunc;

  public:
    inline void operator() (const fix15_short_t * const src,
                            fix15_short_t * const dst,
                            const fix15_short_t src_opacity) const
    {
#ifndef HEAVY_DEBUG
        // Skip tile if it can't affect the backdrop
        const bool skip_empty_src = ! compositefunc.zero_alpha_has_effect;
        if (skip_empty_src && src_opacity == 0) {
            return;
        }
#endif

        // Pixel loop
        fix15_t Rs,Gs,Bs,as, Rb,Gb,Bb,ab, one_minus_ab;
#pragma omp parallel for private(Rs,Gs,Bs,as, Rb,Gb,Bb,ab, one_minus_ab)
        for (unsigned int i = 0; i < BUFSIZE; i += 4)
        {
            // Calculate unpremultiplied source RGB values
            as = src[i+3];
            if (as == 0) {
#ifndef HEAVY_DEBUG
                // Skip pixel if it can't affect the backdrop pixel
                if (skip_empty_src) {
                    continue;
                }
#endif
                // Otherwise just avoid the divide-by-zero by assuming the
                // value before premultiplication was also zero.
                Rs = Gs = Bs = 0;
            }
            else {
                Rs = fix15_short_clamp(fix15_div(src[i+0], as));
                Gs = fix15_short_clamp(fix15_div(src[i+1], as));
                Bs = fix15_short_clamp(fix15_div(src[i+2], as));
            }
#ifdef HEAVY_DEBUG
            assert(Rs <= fix15_one); assert(Rs >= 0);
            assert(Gs <= fix15_one); assert(Gs >= 0);
            assert(Bs <= fix15_one); assert(Bs >= 0);
#endif

            // Calculate unpremultiplied backdrop RGB values
            if (DSTALPHA) {
                ab = dst[i+3];
                if (ab == 0) {
                    Rb = Gb = Bb = 0;
                }
                else {
                    Rb = fix15_short_clamp(fix15_div(dst[i+0], ab));
                    Gb = fix15_short_clamp(fix15_div(dst[i+1], ab));
                    Bb = fix15_short_clamp(fix15_div(dst[i+2], ab));
                }
            }
            else {
                ab = fix15_one;
                Rb = dst[i+0];
                Gb = dst[i+1];
                Bb = dst[i+2];
            }
#ifdef HEAVY_DEBUG
            assert(Rb <= fix15_one); assert(Rb >= 0);
            assert(Gb <= fix15_one); assert(Gb >= 0);
            assert(Bb <= fix15_one); assert(Bb >= 0);
#endif

            // Apply the colour blend functor
            blendfunc(Rs, Gs, Bs, Rb, Gb, Bb);

            // Apply results of the blend in place
            if (DSTALPHA) {
                one_minus_ab = fix15_one - ab;
                Rb = fix15_sumprods(one_minus_ab, Rs, ab, Rb);
                Gb = fix15_sumprods(one_minus_ab, Gs, ab, Gb);
                Bb = fix15_sumprods(one_minus_ab, Bs, ab, Bb);
            }
#ifdef HEAVY_DEBUG
            assert(Rb <= fix15_one); assert(Rb >= 0);
            assert(Gb <= fix15_one); assert(Gb >= 0);
            assert(Bb <= fix15_one); assert(Bb >= 0);
#endif
            // Use the blend result as a source, and composite directly into
            // the destination buffer as premultiplied RGB.
            compositefunc(Rb, Gb, Bb, fix15_mul(as, src_opacity),
                          dst[i+0], dst[i+1], dst[i+2], dst[i+3]);
        }
    }
};


// Abstract interface for tile-sized BufferCombineFunc<>s
//
// This is the interface the Python-facing code uses, one per supported
// tiledsurface (layer) combine mode. Implementations are intended to be
// templated things exposing their CompositeFunc's flags via the
// abstract methods defined in this interface.

class TileDataCombineOp
{
  public:
    virtual void combine_data (const fix15_short_t *src_p,
                               fix15_short_t *dst_p,
                               const bool dst_has_alpha,
                               const float src_opacity) const = 0;
    virtual const char* get_name() const = 0;
    virtual bool zero_alpha_has_effect() const = 0;
    virtual bool can_decrease_alpha() const = 0;
    virtual bool zero_alpha_clears_backdrop() const = 0;
};


// Source Over: place the source over the destination. This implements the
// conventional "basic alpha blending" compositing mode.
// http://www.w3.org/TR/compositing-1/#porterduffcompositingoperators_srcover

class CompositeSourceOver : public CompositeFunc
{
  public:
    inline void operator() (const fix15_t Rs, const fix15_t Gs,
                            const fix15_t Bs, const fix15_t as,
                            fix15_short_t &rb, fix15_short_t &gb,
                            fix15_short_t &bb, fix15_short_t &ab) const
    {
        const fix15_t j = fix15_one - as;
        const fix15_t k = fix15_mul(ab, j);

        rb = fix15_short_clamp(fix15_sumprods(as, Rs, j, rb));
        gb = fix15_short_clamp(fix15_sumprods(as, Gs, j, gb));
        bb = fix15_short_clamp(fix15_sumprods(as, Bs, j, bb));
        ab = fix15_short_clamp(as + k);
    }

    static const bool zero_alpha_has_effect = false;
    static const bool can_decrease_alpha = false;
    static const bool zero_alpha_clears_backdrop = false;
};

// Source Over Spectral WGM:  place the source over the destination
// Similar to paint.  Use weighted geometric mean, upsample to 10 channels
// must use un-premultiplied color and alpha ratios normalized to sum to 1.0

class CompositeSpectralWGM : public CompositeFunc
{
  public:
    inline void operator() (const fix15_t Rs, const fix15_t Gs,
                            const fix15_t Bs, const fix15_t as,
                            fix15_short_t &rb, fix15_short_t &gb,
                            fix15_short_t &bb, fix15_short_t &ab) const
    {
        // psuedo code example:
        // ratio = as / as + (1 - as) * ab;
        // rgb = pow(rgb, ratio) * pow(rgb, (1-ratio));
        // ab = fix15_short_clamp(as + k);
        // rgb = rgb * ab;
    }

    static const bool zero_alpha_has_effect = false;
    static const bool can_decrease_alpha = false;
    static const bool zero_alpha_clears_backdrop = false;
};


// Destination-In: the painted areas make stencil voids. The backdrop shows
// through only within the painted areas of the source.
// http://www.w3.org/TR/compositing-1/#compositingoperators_dstin

class CompositeDestinationIn : public CompositeFunc
{
  public:
    inline void operator() (const fix15_t Rs, const fix15_t Gs,
                            const fix15_t Bs, const fix15_t as,
                            fix15_short_t &rb, fix15_short_t &gb,
                            fix15_short_t &bb, fix15_short_t &ab) const
    {
        rb = fix15_short_clamp(fix15_mul(rb, as));
        gb = fix15_short_clamp(fix15_mul(gb, as));
        bb = fix15_short_clamp(fix15_mul(bb, as));
        ab = fix15_short_clamp(fix15_mul(ab, as));
    }

    static const bool zero_alpha_has_effect = true;
    static const bool can_decrease_alpha = true;
    static const bool zero_alpha_clears_backdrop = true;
};


// Destination-Out: the painted areas work a little like masking fluid or tape,
// or wax resist. The backdrop shows through only outside painted source areas.
// http://www.w3.org/TR/compositing-1/#porterduffcompositingoperators_dstout

class CompositeDestinationOut : public CompositeFunc
{
  public:
    inline void operator() (const fix15_t Rs, const fix15_t Gs,
                            const fix15_t Bs, const fix15_t as,
                            fix15_short_t &rb, fix15_short_t &gb,
                            fix15_short_t &bb, fix15_short_t &ab) const
    {
        const fix15_t j = fix15_one - as;
        rb = fix15_short_clamp(fix15_mul(rb, j));
        gb = fix15_short_clamp(fix15_mul(gb, j));
        bb = fix15_short_clamp(fix15_mul(bb, j));
        ab = fix15_short_clamp(fix15_mul(ab, j));
    }

    static const bool zero_alpha_has_effect = false;
    static const bool can_decrease_alpha = true;
    static const bool zero_alpha_clears_backdrop = false;
};


// Source-Atop: Source which overlaps the destination, replaces the destination.
// Destination is placed elsewhere.
// http://www.w3.org/TR/compositing-1/#porterduffcompositingoperators_srcatop

class CompositeSourceAtop : public CompositeFunc
{
  public:
    inline void operator() (const fix15_t Rs, const fix15_t Gs,
                            const fix15_t Bs, const fix15_t as,
                            fix15_short_t &rb, fix15_short_t &gb,
                            fix15_short_t &bb, fix15_short_t &ab) const
    {
        // W3C spec:
        //   co = as*Cs*ab + ab*Cb*(1-as)
        // where
        //   Cs ∈ {Rs, Gs, Bs}         -- input is non-premultiplied
        //   cb ∈ {rb gb, bb} = ab*Cb  -- output is premultiplied by alpha
        const fix15_t one_minus_as = fix15_one - as;
        const fix15_t ab_mul_as = fix15_mul(as, ab);
        rb = fix15_short_clamp(fix15_sumprods(ab_mul_as, Rs, one_minus_as, rb));
        gb = fix15_short_clamp(fix15_sumprods(ab_mul_as, Gs, one_minus_as, gb));
        bb = fix15_short_clamp(fix15_sumprods(ab_mul_as, Bs, one_minus_as, bb));
        // W3C spec:
        //   ao = as*ab + ab*(1-as)
        //   ao = ab
        // (leave output alpha unchanged)
    }

    static const bool zero_alpha_has_effect = false;
    static const bool can_decrease_alpha = false;
    static const bool zero_alpha_clears_backdrop = false;
};


// Destination-Atop: Destination which overlaps the source replaces the source.
// Source is placed elsewhere.
// http://www.w3.org/TR/compositing-1/#porterduffcompositingoperators_dstatop

class CompositeDestinationAtop : public CompositeFunc
{
  public:
    inline void operator() (const fix15_t Rs, const fix15_t Gs,
                            const fix15_t Bs, const fix15_t as,
                            fix15_short_t &rb, fix15_short_t &gb,
                            fix15_short_t &bb, fix15_short_t &ab) const
    {
        // W3C spec:
        //   co = as*Cs*(1-ab) + ab*Cb*as
        // where
        //   Cs ∈ {Rs, Gs, Bs}         -- input is non-premultiplied
        //   cb ∈ {rb gb, bb} = ab*Cb  -- output is premultiplied by alpha
        const fix15_t one_minus_ab = fix15_one - ab;
        const fix15_t as_mul_one_minus_ab = fix15_mul(as, one_minus_ab);
        rb = fix15_short_clamp(fix15_sumprods(as_mul_one_minus_ab, Rs, as, rb));
        gb = fix15_short_clamp(fix15_sumprods(as_mul_one_minus_ab, Gs, as, gb));
        bb = fix15_short_clamp(fix15_sumprods(as_mul_one_minus_ab, Bs, as, bb));
        // W3C spec:
        //   ao = as*(1-ab) + ab*as
        //   ao = as
        ab = as;
    }

    static const bool zero_alpha_has_effect = true;
    static const bool can_decrease_alpha = true;
    static const bool zero_alpha_clears_backdrop = true;
};


// W3C "Lighter", a.k.a. Porter-Duff "plus", a.k.a. "svg:plus". This just adds
// together corresponding channels of the destination and source.
// Ref: http://www.w3.org/TR/compositing-1/#porterduffcompositingoperators_plus

class CompositeLighter : public CompositeFunc
{
  public:
    inline void operator() (const fix15_t Rs, const fix15_t Gs,
                            const fix15_t Bs, const fix15_t as,
                            fix15_short_t &rb, fix15_short_t &gb,
                            fix15_short_t &bb, fix15_short_t &ab) const
    {
        rb = fix15_short_clamp(fix15_mul(Rs, as) + rb);
        gb = fix15_short_clamp(fix15_mul(Gs, as) + gb);
        bb = fix15_short_clamp(fix15_mul(Bs, as) + bb);
        ab = fix15_short_clamp(ab + as);
    }

    static const bool zero_alpha_has_effect = false;
    static const bool can_decrease_alpha = false;
    static const bool zero_alpha_clears_backdrop = false;
};


#endif //__HAVE_COMPOSITING
