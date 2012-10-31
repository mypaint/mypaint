/* This file is part of MyPaint.
 * Copyright (C) 2012 by Andrew Chadwick <a.t.chadwick@gmail.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */


/*
Where possible, work in premultiplied-alpha and use the terminology from
the W3C specifications for compositing modes available at

  http://www.w3.org/TR/SVGCompositing/
  http://dvcs.w3.org/hg/FXTF/rawfile/tip/compositing/index.html

Namely,

  Sc - source (top) layer channel value, non-premultiplied
  Sa - source (top) layer alpha: src_p[3] x opac
  Sca - source (top) layer channel value, premult by Da: src_p[c] x opac
  Dc - destination layer channel value, non-premultiplied
  Da - destination layer alpha: dst_p[3]
  Dca - destination layer channel value, premult by Da: dst_p[c]

Values from src_p[] arguments are almost(?) always multiplied by the layer
opacity, opac, before use as above.

For now, this file is included twice, once with COMPOSITE_MODE_RGBA defined
and once without it. In the latter case, assume Da=fix15_one, and don't write
to the destination alpha.

*/


// It's always better to use inline functions rather than than fumble around
// with scaling factors: the code will be much cleaner. It's OK to define
// optimal funcs for special cases like a sum of two products.
#include "fix15.hpp"




inline void
#ifdef COMPOSITE_MODE_RGBA
rgba_composite_src_over_rgba
#else
rgba_composite_src_over_rgbu
#endif
    (const fix15_short_t src_p[],
     fix15_short_t dst_p[],
     const fix15_short_t opac)
{
    // http://www.w3.org/TR/SVGCompositing/#comp-op-src-over
    const fix15_t Sa = fix15_mul(src_p[3], opac);
    dst_p[0] = fix15_sumprods( src_p[0], opac,   fix15_one-Sa, dst_p[0] );
    dst_p[1] = fix15_sumprods( src_p[1], opac,   fix15_one-Sa, dst_p[1] );
    dst_p[2] = fix15_sumprods( src_p[2], opac,   fix15_one-Sa, dst_p[2] );
#ifdef COMPOSITE_MODE_RGBA
    const fix15_t Da = dst_p[3];
    dst_p[3] = Sa + Da - fix15_mul(Sa, Da);
#endif
}


inline void
#ifdef COMPOSITE_MODE_RGBA
rgba_composite_multiply_rgba
#else
rgba_composite_multiply_rgbu
#endif
    (const fix15_short_t src_p[],
     fix15_short_t dst_p[],
     const fix15_short_t opac)
{
    // http://www.w3.org/TR/SVGCompositing/#comp-op-multiply
    const fix15_t Sa = fix15_mul(src_p[3], opac);
    // Dca' = Sca*Dca + Sca*(1 - Da) + Dca*(1 - Sa)
    //      = Dca * (1 + Sca - Sa)                   ; if Da == 1
    //      = Dca * (1 + Sca - Sa) + Sca*(1 - Da)    ; otherwise
    const fix15_t Sca0 = fix15_mul(src_p[0], opac);
    const fix15_t Sca1 = fix15_mul(src_p[1], opac);
    const fix15_t Sca2 = fix15_mul(src_p[2], opac);
    dst_p[0] = fix15_mul(dst_p[0], fix15_one + Sca0 - Sa);
    dst_p[1] = fix15_mul(dst_p[1], fix15_one + Sca1 - Sa);
    dst_p[2] = fix15_mul(dst_p[2], fix15_one + Sca2 - Sa);
#ifdef COMPOSITE_MODE_RGBA
    const fix15_t Da = dst_p[3];
    dst_p[0] += fix15_mul(Sca0, fix15_one - Da);
    dst_p[1] += fix15_mul(Sca1, fix15_one - Da);
    dst_p[2] += fix15_mul(Sca2, fix15_one - Da);
    dst_p[3] = Sa + Da - fix15_mul(Sa, Da);
#endif
}


inline void
#ifdef COMPOSITE_MODE_RGBA
rgba_composite_screen_rgba
#else
rgba_composite_screen_rgbu
#endif
    (const fix15_short_t src_p[],
     fix15_short_t dst_p[],
     const fix15_short_t opac)
{
    // http://www.w3.org/TR/SVGCompositing/#comp-op-multiply
    // Dca' = (Sca*Da + Dca*Sa - Sca*Dca) + Sca*(1 - Da) + Dca*(1 - Sa)
    //      = Sca + Dca - Sca*Dca

    const fix15_t Sca0 = fix15_mul(src_p[0], opac);
    const fix15_t Sca1 = fix15_mul(src_p[1], opac);
    const fix15_t Sca2 = fix15_mul(src_p[2], opac);
    dst_p[0] += Sca0 - fix15_mul(Sca0, dst_p[0]);
    dst_p[1] += Sca1 - fix15_mul(Sca1, dst_p[1]);
    dst_p[2] += Sca2 - fix15_mul(Sca2, dst_p[2]);
#ifdef COMPOSITE_MODE_RGBA
    const fix15_t Sa = fix15_mul(src_p[3], opac);
    const fix15_t Da = fix15_mul(dst_p[3], opac);
    dst_p[3] = Sa + Da - fix15_mul(Sa, Da);
#endif
}


inline void
#ifdef COMPOSITE_MODE_RGBA
rgba_composite_overlay_rgba
#else
rgba_composite_overlay_rgbu
#endif
    (const fix15_short_t src_p[],
     fix15_short_t dst_p[],
     const fix15_short_t opac)
{
    const fix15_t Sa = fix15_mul(src_p[3], opac);
    const fix15_t one_minus_Sa = fix15_one - Sa;
#ifdef COMPOSITE_MODE_RGBA
    const fix15_t Da = dst_p[3];
    const fix15_t one_minus_Da = fix15_one - Da;
    const fix15_t SaDa = fix15_mul(Sa, Da);
#else
    const fix15_t Da = fix15_one;
#endif
    // http://www.w3.org/TR/SVGCompositing/#comp-op-overlay
    // if 2 * Dca <= Da
    //   Dca' = 2*Sca*Dca + Sca*(1 - Da) + Dca*(1 - Sa)
    // otherwise
    //   Dca' = Sa*Da - 2*(Da - Dca)*(Sa - Sca) + Sca*(1 - Da) + Dca*(1 - Sa)
    //        = Sca*(1 + Da) + Dca*(1 + Sa) - 2*Dca*Sca - Da*Sa
    for (int c=0; c<3; c++) {
        const fix15_t Dca = dst_p[c];
        const fix15_t twoDca = Dca * 2;
        const fix15_t Sca = fix15_mul(src_p[c], opac);
        fix15_t Dca_out = 0;
        if (twoDca <= Da) {
            Dca_out = fix15_sumprods( twoDca, Sca,  Dca, one_minus_Sa );
#ifdef COMPOSITE_MODE_RGBA
            // (1-Da) != 0
            Dca_out += fix15_mul(Sca, one_minus_Da);
#endif //COMPOSITE_MODE_RGBA
        }
        else {
#ifdef COMPOSITE_MODE_RGBA
            Dca_out = fix15_sumprods(Sca, fix15_one+Da,  Dca, fix15_one+Sa)
                    - fix15_mul(twoDca, Sca)
                    - SaDa;
#else
            // Da == 1
            Dca_out = Sca * 2
                    + fix15_mul(Dca, fix15_one+Sa)
                    - fix15_mul(twoDca, Sca)
                    - Sa;
#endif //COMPOSITE_MODE_RGBA
        }
        dst_p[c] = fix15_short_clamp(Dca_out);
#ifdef HEAVY_DEBUG
        assert(dst_p[c] <= (1<<15));
        assert(src_p[c] <= (1<<15));
#endif //HEAVY_DEBUG
    }

#ifdef COMPOSITE_MODE_RGBA
    dst_p[3] = Sa + Da - SaDa;
#ifdef HEAVY_DEBUG
    assert(src_p[0] <= src_p[3]);
    assert(dst_p[0] <= dst_p[3]);
    assert(src_p[1] <= src_p[3]);
    assert(dst_p[1] <= dst_p[3]);
    assert(src_p[2] <= src_p[3]);
    assert(dst_p[2] <= dst_p[3]);
#endif //HEAVY_DEBUG
#endif //COMPOSITE_MODE_RGBA
}


inline void
#ifdef COMPOSITE_MODE_RGBA
rgba_composite_hard_light_rgba
#else
rgba_composite_hard_light_rgbu
#endif
    (const fix15_short_t src_p[],
     fix15_short_t dst_p[],
     const fix15_short_t opac)
{
    const fix15_t Sa = fix15_mul(src_p[3], opac);
    const fix15_t one_minus_Sa = fix15_one - Sa;
#ifdef COMPOSITE_MODE_RGBA
    const fix15_t Da = dst_p[3];
    const fix15_t one_minus_Da = fix15_one - Da;
    const fix15_t SaDa = fix15_mul(Sa, Da);
#endif
    // From http://www.w3.org/TR/SVGCompositing/#comp-op-hard-light --
    // if 2 * Sca <= Da
    //   Dca' = 2*Sca*Dca + Sca*(1 - Da) + Dca*(1 - Sa)
    // otherwise
    //   Dca' = Sa*Da - 2*(Da - Dca)*(Sa - Sca) + Sca*(1 - Da) + Dca*(1 - Sa)
    //        = Sca*(1 + Da) + Dca*(1 + Sa) - 2*Dca*Sca - Da*Sa
    //
    // Identical to Overlay, but with a different test.
    for (int c=0; c<3; c++) {
        const fix15_t Dca = dst_p[c];
        const fix15_t twoDca = Dca * 2;
        const fix15_t Sca = fix15_mul(src_p[c], opac);
        const fix15_t twoSca = Sca * 2;
        fix15_t Dca_out = 0;
        if (twoSca <= Sa) {
            Dca_out = fix15_mul(twoDca, Sca)
                    + fix15_mul(Dca, one_minus_Sa);
#ifdef COMPOSITE_MODE_RGBA
            // (1-Da) != 0
            Dca_out += fix15_mul(Sca, one_minus_Da);
#endif //COMPOSITE_MODE_RGBA
        }
        else {
            // Dca' = Sca*(1 + Da) + Dca*(1 + Sa) - 2*Dca*Sca - Da*Sa
#ifdef COMPOSITE_MODE_RGBA
            Dca_out = fix15_mul(Sca, fix15_one + Da)
                    + fix15_mul(Dca, fix15_one + Sa)
                    - fix15_mul(twoDca, Sca)
                    - SaDa;
#else
            // Da == 1
            Dca_out = twoSca
                    + fix15_mul(Dca, fix15_one + Sa)
                    - fix15_mul(twoDca, Sca)
                    - Sa;
#endif //COMPOSITE_MODE_RGBA
        }
        dst_p[c] = fix15_short_clamp(Dca_out);
#ifdef HEAVY_DEBUG
        assert(dst_p[c] <= fix15_one);
        assert(src_p[c] <= fix15_one);
#endif //HEAVY_DEBUG
    }

#ifdef COMPOSITE_MODE_RGBA
    dst_p[3] = Sa + Da - SaDa;
#ifdef HEAVY_DEBUG
    assert(src_p[0] <= src_p[3]);
    assert(dst_p[0] <= dst_p[3]);
    assert(src_p[1] <= src_p[3]);
    assert(dst_p[1] <= dst_p[3]);
    assert(src_p[2] <= src_p[3]);
    assert(dst_p[2] <= dst_p[3]);
#endif //HEAVY_DEBUG
#endif //COMPOSITE_MODE_RGBA
}


inline void
#ifdef COMPOSITE_MODE_RGBA
rgba_composite_soft_light_rgba
#else
rgba_composite_soft_light_rgbu
#endif
    (const fix15_short_t src_p[],
     fix15_short_t dst_p[],
     const fix15_short_t opac)
{
    /* <URI:https://dvcs.w3.org/hg/FXTF/rawfile/tip/compositing/index.html
            #blendingsoftlight > */

    // Leave the backdrop alone if the source is fully transparent
    const fix15_t a_s = fix15_mul(src_p[3], opac);
    if (a_s == 0) {
        return;
    }

    // If the backdrop (dst) is fully transparent, it becomes the
    // source times opacity.
#ifdef COMPOSITE_MODE_RGBA
    const fix15_t a_b = dst_p[3];
    if (a_b == 0) {
        dst_p[0] = fix15_short_clamp(fix15_mul(opac, src_p[0]));
        dst_p[1] = fix15_short_clamp(fix15_mul(opac, src_p[1]));
        dst_p[2] = fix15_short_clamp(fix15_mul(opac, src_p[2]));
        dst_p[3] = a_s;
        return;
    }
#else
    const fix15_t a_b = fix15_one;
#endif //COMPOSITE_MODE_RGBA

#ifdef HEAVY_DEBUG
    // Underflow is a possibility here
    assert(a_b <= fix15_one);
    assert(a_s <= fix15_one);
    // Confirm we guarded against divisions by zero below
    assert(a_b > 0);
    assert(a_s > 0);
#endif

    for (int i=0; i<3; ++i) {
        // De-premultiplied input components from premultiplied
        const fix15_t aC_s = fix15_mul(opac, src_p[i]);
        const fix15_t aC_b = dst_p[i];
        const fix15_t C_s = fix15_div(aC_s, a_s);
        const fix15_t C_b = fix15_div(aC_b, a_b);

        // The guts of it, a blending function B(C_b, C_s) whose output is
        // used as the input to a regular src-over operation.
        fix15_t B = 0;
        const fix15_t two_C_s = C_s << 1;
        if (two_C_s <= fix15_one) {  // i.e. C_s < 0.5
            B = fix15_one - fix15_mul(fix15_one - two_C_s,
                                      fix15_one - C_b);
            B = fix15_mul(B, C_b);
        }
        else {
            fix15_t D = 0;
            const fix15_t four_C_b = C_b << 2;
            if (four_C_b <= fix15_one) {
                const fix15_t C_b_squared = fix15_mul(C_b, C_b);
                D = four_C_b; /* which is always greater than... */
                D += 16 * fix15_mul(C_b_squared, C_b);
                D -= 12 * C_b_squared;
                /* ... in the range 0 <= C_b <= 0.25 */
            }
            else {
                D = fix15_sqrt(C_b);
            }
#ifdef HEAVY_DEBUG
            /* Guard against underflows */
            assert(2*C_s > fix15_one);
            assert(D >= C_b);
#endif
            B = C_b + fix15_mul(2*C_s - fix15_one /* 2*C_s > 1 */,
                                D - C_b           /* D >= C_b */  );
        }

        // Composite a premultiplied output component as src-over.
        fix15_t aC_o = fix15_mul(fix15_one - a_b, aC_s)
                     + fix15_mul(fix15_one - a_s, aC_b)
                     + fix15_mul(B, fix15_mul(a_s, a_b));
        dst_p[i] = fix15_short_clamp(aC_o);
    }

#ifdef COMPOSITE_MODE_RGBA
    dst_p[3] = fix15_short_clamp(a_s + a_b - fix15_mul(a_s, a_b));
#ifdef HEAVY_DEBUG
    assert(src_p[0] <= src_p[3]);
    assert(dst_p[0] <= dst_p[3]);
    assert(src_p[1] <= src_p[3]);
    assert(dst_p[1] <= dst_p[3]);
    assert(src_p[2] <= src_p[3]);
    assert(dst_p[2] <= dst_p[3]);
#endif //HEAVY_DEBUG
#endif
}


inline void
#ifdef COMPOSITE_MODE_RGBA
rgba_composite_color_dodge_rgba
#else
rgba_composite_color_dodge_rgbu
#endif
    (const fix15_short_t src_p[],
     fix15_short_t dst_p[],
     const fix15_short_t opac)
{
    // http://www.w3.org/TR/SVGCompositing/#comp-op-color-dodge
    const fix15_t Sa = fix15_mul(src_p[3], opac);
    const fix15_t one_minus_Sa = fix15_one - Sa;
#ifdef COMPOSITE_MODE_RGBA
    const fix15_t Da = dst_p[3];
    const fix15_t one_minus_Da = fix15_one - Da;
    const fix15_t SaDa = fix15_mul(Sa, Da);
#else
    const fix15_t SaDa = Sa;
#endif

    for (int c=0; c<3; c++) {
        const fix15_t Sca = fix15_mul(src_p[c], opac);
        const fix15_t Dca = dst_p[c];
        fix15_t Dca_out = 0;
        if (Sca >= Sa) {
            if (Dca == 0) {
                // Sca == Sa and Dca == 0
                //  Dca' = Sca*(1 - Da) + Dca*(1 - Sa)
                //       = Sca*(1 - Da)
#ifdef COMPOSITE_MODE_RGBA
                Dca_out = fix15_mul(Sca, one_minus_Da);
#else
                Dca_out = 0;
#endif //COMPOSITE_MODE_RGBA
            }
            else {
                // otherwise if Sca == Sa and Dca > 0
                //  Dca' = Sa*Da + Sca*(1 - Da) + Dca*(1 - Sa)
                //       = Sca*Da + Sca*(1 - Da) + Dca*(1 - Sa)
                //       = Sca*(Da + 1 - Da) + Dca*(1 - Sa)
                //       = Sca + Dca*(1 - Sa)
                Dca_out = Sca + fix15_mul(Dca, one_minus_Sa);
            }
        }
        else {
            // Sca < Sa
            //    Dca' = Sa*Da * min(1, m) + Sca*(1 - Da) + Dca*(1 - Sa)
            // Where
            //       m = Dca/Da * Sa/(Sa - Sca)
            //         = (Dca*Sa) / (Da*(Sa - Sca))
            fix15_t m = 0;
#ifdef COMPOSITE_MODE_RGBA
            if (Da != 0) {
                m = fix15_div(fix15_mul(Dca, Sa), fix15_mul(Da, Sa - Sca));
            }
            Dca_out = fix15_sumprods(Sca, one_minus_Da, Dca, one_minus_Sa);
#else
            // Da == 1
            m = fix15_div(fix15_mul(Dca, Sa), Sa - Sca);
            Dca_out = fix15_mul(Dca, one_minus_Sa);
#endif
            if (m < fix15_one) {
                Dca_out += fix15_mul(SaDa, m);
            }
            else {
                Dca_out += SaDa;
            }
        }
        dst_p[c] = fix15_short_clamp(Dca_out);
#ifdef HEAVY_DEBUG
        assert(dst_p[c] <= fix15_one);
        assert(src_p[c] <= fix15_one);
#endif
    }

#ifdef COMPOSITE_MODE_RGBA
    // Da'  = Sa + Da - Sa*Da
    dst_p[3] = fix15_short_clamp(Sa + Da - SaDa);
#ifdef HEAVY_DEBUG
    assert(src_p[0] <= src_p[3]);
    assert(dst_p[0] <= dst_p[3]);
    assert(src_p[1] <= src_p[3]);
    assert(dst_p[1] <= dst_p[3]);
    assert(src_p[2] <= src_p[3]);
    assert(dst_p[2] <= dst_p[3]);
#endif //HEAVY_DEBUG
#endif //COMPOSITE_MODE_RGBA

#ifdef HEAVY_DEBUG
    assert(dst_p[3] <= fix15_one);
    assert(src_p[3] <= fix15_one);
#endif
}



inline void
#ifdef COMPOSITE_MODE_RGBA
rgba_composite_color_burn_rgba
#else
rgba_composite_color_burn_rgbu
#endif
    (const fix15_short_t src_p[],
     fix15_short_t dst_p[],
     const fix15_short_t opac)
{
    // http://www.w3.org/TR/SVGCompositing/#comp-op-color-burn
    const fix15_t Sa = fix15_mul(src_p[3], opac);
    const fix15_t one_minus_Sa = fix15_one - Sa;
#ifdef COMPOSITE_MODE_RGBA
    const fix15_t Da = dst_p[3];
    const fix15_t one_minus_Da = fix15_one - Da;
#else
    const fix15_t Da = fix15_one;
#endif
    for (int c=0; c<3; c++) {
        const fix15_t Sca = fix15_mul(src_p[c], opac);
        const fix15_t Dca = dst_p[c];
        if (Sca == 0) {
            //if Sca == 0 and Dca == Da
            //  Dca' = Sa*Da + Sca*(1 - Da) + Dca*(1 - Sa)
            //       = Sa*Dca + Dca*(1 - Sa)
            //       = Sa*Dca + Dca - Sa*Dca
            //       = Dca
            if (Dca != Da) {
                //otherwise (when Sca == 0)
                //  Dca' = Sca*(1 - Da) + Dca*(1 - Sa)
                //       = Dca*(1 - Sa)
                dst_p[c] = fix15_short_clamp(fix15_mul(Dca, one_minus_Sa));
            }
        }
        else {
#ifdef HEAVY_DEBUG
            assert(Sca <= fix15_one);
            assert(Sca > 0);
#endif
            //otherwise if Sca > 0
            //  let i = Sca*(1 - Da) + Dca*(1 - Sa)
            //  let m = (1 - Dca/Da) * Sa/Sca
            //
            //  Dca' = Sa*Da - Sa*Da * min(1, (1 - Dca/Da) * Sa/Sca) + i
            //       = Sa*Da * (1 - min(1, (1 - Dca/Da) * Sa/Sca)) + i

#ifdef COMPOSITE_MODE_RGBA
            fix15_t res = fix15_sumprods(Sca, one_minus_Da, Dca, one_minus_Sa);
            if (Da > 0) {
                const fix15_t m = fix15_div(fix15_mul(
                                              fix15_one - fix15_div(Dca, Da),
                                              Sa),
                                            Sca);
                if (m < fix15_one) {
                    res += fix15_mul(fix15_mul(Sa, Da), fix15_one - m);
                }
            }
#else
            fix15_t res = fix15_mul(Dca, one_minus_Sa);
            const fix15_t m = fix15_div(fix15_mul(
                                          fix15_one - Dca,
                                          Sa),
                                        Sca);
            if (m < fix15_one) {
                res += fix15_mul(Sa, fix15_one - m);
            }
#endif
            dst_p[c] = fix15_short_clamp(res);
        }
#ifdef HEAVY_DEBUG
        assert(dst_p[c] <= fix15_one);
        assert(src_p[c] <= fix15_one);
#endif
    }

#ifdef COMPOSITE_MODE_RGBA
    // Da'  = Sa + Da - Sa*Da
    dst_p[3] = fix15_short_clamp(Sa + Da
                                 - fix15_mul(Sa, Da));
#ifdef HEAVY_DEBUG
    assert(src_p[0] <= src_p[3]);
    assert(dst_p[0] <= dst_p[3]);
    assert(src_p[1] <= src_p[3]);
    assert(dst_p[1] <= dst_p[3]);
    assert(src_p[2] <= src_p[3]);
    assert(dst_p[2] <= dst_p[3]);
#endif //HEAVY_DEBUG
#endif //COMPOSITE_MODE_RGBA

#ifdef HEAVY_DEBUG
    assert(dst_p[3] <= fix15_one);
    assert(src_p[3] <= fix15_one);
#endif
}



// Non-separable blend modes.
// http://dvcs.w3.org/hg/FXTF/rawfile/tip/compositing/index.html
// Same as the ones in Cairo, and in the PDF specs.


#ifndef __HAVE_NONSEP_MAPFUNC
#define __HAVE_NONSEP_MAPFUNC
typedef void (*_nonseparable_mapfunc) (const uint16_t /* src red in */,
                                       const uint16_t /* src green in */,
                                       const uint16_t /* src blue in */,
                                       uint16_t * /* dst red in/out */,
                                       uint16_t * /* dst green in/out */,
                                       uint16_t * /* dst blue in/out */);
#endif // __HAVE_NONSEP_MAPFUNC



static inline void
#ifdef COMPOSITE_MODE_RGBA
_rgba_composite_nonseparable_over_rgba
#else
_rgba_composite_nonseparable_over_rgbu
#endif
    (const uint16_t src_p[],
     uint16_t dst_p[],
     const uint16_t opac,
     const _nonseparable_mapfunc mapfunc)
{
    uint16_t src_r, src_g, src_b;
    src_r = src_g = src_b = 0;
    const uint16_t src_a = src_p[3];
    if (src_a == 0)
        return;

    // De-premult
    src_r = ((1<<15)*((uint32_t)src_p[0])) / src_a;
    src_g = ((1<<15)*((uint32_t)src_p[1])) / src_a;
    src_b = ((1<<15)*((uint32_t)src_p[2])) / src_a;

    // Create a temporary "source" colour based on dst_p, colorized in
    // the desired way by src_p.

    uint16_t tmp_p[4] = { dst_p[0], dst_p[1], dst_p[2], src_a };
    mapfunc ( src_r, src_g, src_b,
              &tmp_p[0], &tmp_p[1], &tmp_p[2] );

    // Re-premult
    tmp_p[0] = ((uint32_t) tmp_p[0]) * src_a / (1<<15);
    tmp_p[1] = ((uint32_t) tmp_p[1]) * src_a / (1<<15);
    tmp_p[2] = ((uint32_t) tmp_p[2]) * src_a / (1<<15);

    // Combine it in the normal way with the destination layer.
#ifdef COMPOSITE_MODE_RGBA
    rgba_composite_src_over_rgba (tmp_p, dst_p, opac);
#else
    rgba_composite_src_over_rgbu (tmp_p, dst_p, opac);
#endif
}



#ifndef __HAVE_SVGFX_BLENDS
#define __HAVE_SVGFX_BLENDS

// Luma/luminance coefficients, from the spec linked above as of Mercurial
// revision fc58b9389b07, dated Thu Jul 26 07:27:58 2012. They're similar, but
// not identical to, the ones defined in Rec. ITU-R BT.601-7 Section 2.5.1.

static const float SVGFX_LUM_R_COEFF = 0.3;
static const float SVGFX_LUM_G_COEFF = 0.59;
static const float SVGFX_LUM_B_COEFF = 0.11;


// Returns the luma/luminance of an RGB triple, expressed as scaled ints.

#define svgfx_lum(r,g,b) \
   (  (r) * (uint16_t)(SVGFX_LUM_R_COEFF * (1<<15)) \
    + (g) * (uint16_t)(SVGFX_LUM_G_COEFF * (1<<15)) \
    + (b) * (uint16_t)(SVGFX_LUM_B_COEFF * (1<<15))  )


#ifndef MIN3
#define MIN3(a,b,c) ( (a)<(b) ? MIN((a), (c)) : MIN((b), (c)) )
#endif
#ifndef MAX3
#define MAX3(a,b,c) ( (a)>(b) ? MAX((a), (c)) : MAX((b), (c)) )
#endif



// Sets the target's luma/luminance to that of the input, retaining its hue
// angle and clipping the saturation if necessary.
//
//
// All params are scaled ints having factor 2**-15, and must not store
// premultiplied alpha.


inline void
svgfx_blend_color(const uint16_t r0,
                  const uint16_t g0,
                  const uint16_t b0,
                  uint16_t *r1,
                  uint16_t *g1,
                  uint16_t *b1)
{
    // Spec: SetLum()
    // Colours potentially can go out of band to both sides, hence the
    // temporary representation inflation.
    const uint16_t lum1 = svgfx_lum(*r1, *g1, *b1) / (1<<15);
    const uint16_t lum0 = svgfx_lum(r0, g0, b0) / (1<<15);
    const int16_t diff = lum1 - lum0;
    int32_t r = r0 + diff;
    int32_t g = g0 + diff;
    int32_t b = b0 + diff;

    // Spec: ClipColor()
    // Trim out of band values, retaining lum.
    int32_t lum = svgfx_lum(r, g, b) / (1<<15);
    int32_t cmin = MIN3(r, g, b);
    int32_t cmax = MAX3(r, g, b);

    if (cmin < 0) {
        r = lum + (((r - lum) * lum) / (lum - cmin));
        g = lum + (((g - lum) * lum) / (lum - cmin));
        b = lum + (((b - lum) * lum) / (lum - cmin));
    }
    if (cmax > (1<<15)) {
        r = lum + (((r - lum) * ((1<<15)-lum)) / (cmax - lum));
        g = lum + (((g - lum) * ((1<<15)-lum)) / (cmax - lum));
        b = lum + (((b - lum) * ((1<<15)-lum)) / (cmax - lum));
    }
#ifdef HEAVY_DEBUG
    assert((0 <= r) && (r <= (1<<15)));
    assert((0 <= g) && (g <= (1<<15)));
    assert((0 <= b) && (b <= (1<<15)));
#endif

    *r1 = r;
    *g1 = g;
    *b1 = b;
}



// The two modes are defined to be symmetrical (whether or not that's really a
// good idea artistically). The Luminosity blend mode suffers from contrast
// issues, but it's likely to be used less by artists.

inline void
svgfx_blend_luminosity(const uint16_t r0,
                       const uint16_t g0,
                       const uint16_t b0,
                       uint16_t *r1,
                       uint16_t *g1,
                       uint16_t *b1)
{
    uint16_t r = r0;
    uint16_t g = g0;
    uint16_t b = b0;
    svgfx_blend_color(*r1, *g1, *b1, &r, &g, &b);
    *r1 = r;
    *g1 = g;
    *b1 = b;
}


#endif // __HAVE_SVGFX_BLENDS



static inline void
#ifdef COMPOSITE_MODE_RGBA
rgba_composite_color_rgba
#else
rgba_composite_color_rgbu
#endif
    (const uint16_t src_p[],
     uint16_t dst_p[],
     const uint16_t opac)
{
#ifdef COMPOSITE_MODE_RGBA
_rgba_composite_nonseparable_over_rgba
#else
_rgba_composite_nonseparable_over_rgbu
#endif
                                      (src_p, dst_p, opac,
                                       svgfx_blend_color);
}


static inline void
#ifdef COMPOSITE_MODE_RGBA
rgba_composite_luminosity_rgba
#else
rgba_composite_luminosity_rgbu
#endif
    (const uint16_t src_p[],
     uint16_t dst_p[],
     const uint16_t opac)
{
#ifdef COMPOSITE_MODE_RGBA
_rgba_composite_nonseparable_over_rgba
#else
_rgba_composite_nonseparable_over_rgbu
#endif
                                      (src_p, dst_p, opac,
                                       svgfx_blend_luminosity);
}


