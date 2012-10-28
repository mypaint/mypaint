/* This file is part of MyPaint.
 * Copyright (C) 2012 by Andrew Chadwick <a.t.chadwick@gmail.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */


inline void
#ifdef COMPOSITE_MODE_RGBA
rgba_composite_src_over_rgba
#else
rgba_composite_src_over_rgbu
#endif
    (const uint16_t src_p[],
     uint16_t dst_p[],
     const uint16_t opac)
{
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

#ifdef COMPOSITE_MODE_RGBA
    // Da' = Sa*Da + Sa*(1 - Da) + Da*(1 - Sa)
    //     = (Sa*Da + Sa - Sa*Da) + Da*(1 - Sa)
    //     = Sa + Da*(1 - Sa)
    //
    dst_p[3] = src_alpha + ((one_minus_src_alpha*dst_p[3]) >> 15);
#endif
}


inline void
#ifdef COMPOSITE_MODE_RGBA
rgba_composite_multiply_rgba
#else
rgba_composite_multiply_rgbu
#endif
    (const uint16_t src_p[],
     uint16_t dst_p[],
     const uint16_t opac)
{
    // Dca' = Sca*Dca + Sca*(1 - Da) + Dca*(1 - Sa)
    //
    // If Da == 1, which is the case in RGBU mode, this becomes
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
    dst_p[0] = ((uint32_t)src_col0*dst_p[0]+one_minus_src_alpha*dst_p[0])>>15;
    dst_p[1] = ((uint32_t)src_col1*dst_p[1]+one_minus_src_alpha*dst_p[1])>>15;
    dst_p[2] = ((uint32_t)src_col2*dst_p[2]+one_minus_src_alpha*dst_p[2])>>15;
#ifdef COMPOSITE_MODE_RGBA
    // Sca*(1 - Da) != 0, add it in
    const uint32_t one_minus_dst_alpha = (1<<15) - dst_p[3];
    dst_p[0] += (((uint32_t)src_col0 * one_minus_dst_alpha) >> 15);
    dst_p[1] += (((uint32_t)src_col1 * one_minus_dst_alpha) >> 15);
    dst_p[2] += (((uint32_t)src_col2 * one_minus_dst_alpha) >> 15);

    // Da'  = Sa*Da + Sa*(1 - Da) + Da*(1 - Sa)
    //      = (Sa*Da + Sa - Sa*Da) + Da*(1 - Sa)
    //      = Sa + (1 - Sa)*Da
    dst_p[3] = src_alpha + ((one_minus_src_alpha*dst_p[3]) >> 15);
#endif
}


inline void
#ifdef COMPOSITE_MODE_RGBA
rgba_composite_screen_rgba
#else
rgba_composite_screen_rgbu
#endif
    (const uint16_t src_p[],
     uint16_t dst_p[],
     const uint16_t opac)
{
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
    dst_p[0] = (col0 - ((uint32_t)src_col0*dst_p[0])) >> 15;
    dst_p[1] = (col1 - ((uint32_t)src_col1*dst_p[1])) >> 15;
    dst_p[2] = (col2 - ((uint32_t)src_col2*dst_p[2])) >> 15;
#ifdef COMPOSITE_MODE_RGBA
    // Da'  = Sa + Da - Sa*Da
    const uint32_t src_alpha = ((uint32_t)src_p[3] * opac) >> 15;
    dst_p[3] =   src_alpha + (uint32_t)dst_p[3]
             - ((src_alpha * (uint32_t)dst_p[3]) >> 15);
#endif
}


inline void
#ifdef COMPOSITE_MODE_RGBA
rgba_composite_overlay_rgba
#else
rgba_composite_overlay_rgbu
#endif
    (const uint16_t src_p[],
     uint16_t dst_p[],
     const uint16_t opac)
{
    const uint32_t Sa = ((uint32_t)src_p[3] * opac)>>15;
    const uint32_t one_minus_Sa = (1<<15) - Sa;
#ifdef COMPOSITE_MODE_RGBA
    const uint32_t Da = dst_p[3];
    const uint32_t one_minus_Da = (1<<15) - Da;
    const uint32_t SaDa = (Sa * Da) >> 15;
#else
    const uint32_t Da = 1<<15;
#endif
    // From http://www.w3.org/TR/SVGCompositing/#comp-op-overlay --
    // if 2 * Dca <= Da
    //   Dca' = 2*Sca*Dca + Sca*(1 - Da) + Dca*(1 - Sa)
    // otherwise
    //   Dca' = Sa*Da - 2*(Da - Dca)*(Sa - Sca) + Sca*(1 - Da) + Dca*(1 - Sa)
    //        = Sca*(1 + Da) + Dca*(1 + Sa) - 2*Dca*Sca - Da*Sa
    for (int c=0; c<3; c++) {
        const uint32_t Dca = dst_p[c];
        const uint32_t twoDca = Dca * 2;
        const uint32_t Sca = ((uint32_t)src_p[c] * opac)>>15;
        uint32_t Dca_out = 0;
        if (twoDca <= Da) {
            Dca_out = ((twoDca * Sca)>>15)
                    + ((Dca * one_minus_Sa)>>15);
#ifdef COMPOSITE_MODE_RGBA
            // (1-Da) != 0
            Dca_out += ((Sca * one_minus_Da)>>15);
#endif //COMPOSITE_MODE_RGBA
        }
        else {
#ifdef COMPOSITE_MODE_RGBA
            Dca_out = ((Sca * ((1<<15) + Da))>>15)
                    + ((Dca * ((1<<15) + Sa))>>15)
                    - ((twoDca * Sca) >> 15)
                    - SaDa;
#else
            // Da == 1
            Dca_out = (Sca * 2)
                    + ((Dca * ((1<<15) + Sa))>>15)
                    - ((twoDca * Sca) >> 15)
                    - Sa;
#endif //COMPOSITE_MODE_RGBA
        }
        dst_p[c] = CLAMP(Dca_out, 0, (1<<15));
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
rgba_composite_color_dodge_rgba
#else
rgba_composite_color_dodge_rgbu
#endif
    (const uint16_t src_p[],
     uint16_t dst_p[],
     const uint16_t opac)
{
    const uint32_t src_alpha = ((uint32_t)src_p[3]*opac)>>15;
    const uint32_t one_minus_src_alpha = (1<<15) - src_alpha;
#ifdef COMPOSITE_MODE_RGBA
    const uint32_t dst_alpha = dst_p[3];
    const uint32_t one_minus_dst_alpha = (1<<15) - dst_alpha;
#endif

    for (int c=0; c<3; c++) {
        const uint32_t s = ((uint32_t)src_p[c]*opac)>>15;
        const uint32_t d = dst_p[c];
        const uint32_t src_alpha_minus_src = src_alpha - s;
        if (src_alpha_minus_src == 0 && d == 0) {
            // Sca == Sa and Dca == 0
            //  Dca' = Sca*(1 - Da) + Dca*(1 - Sa)
            //       = Sca*(1 - Da)
#ifdef COMPOSITE_MODE_RGBA
            dst_p[c] = CLAMP((s * one_minus_dst_alpha) >> 15, 0, (1<<15));
#else
            dst_p[c] = 0;
#endif
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
#ifdef COMPOSITE_MODE_RGBA
            if (dst_times_src_alpha_B30 > (dst_alpha * src_alpha_minus_src))
#else
            if (dst_times_src_alpha_B30 > (src_alpha_minus_src << 15))
#endif
            {
                // min(...)==1
                //    Dca' = Sa * Da * min(...) + Sca*(1 - Da) + Dca*(1 - Sa)
                //    Dca' = Sa * Da + Sca*(1 - Da) + Dca*(1 - Sa)
                dst_p[c] = CLAMP(
                      (  (d * one_minus_src_alpha) // B30
#ifdef COMPOSITE_MODE_RGBA
                       + (s * one_minus_dst_alpha) // B30
                       + (src_alpha * dst_alpha)   // B30
#else
                    // + (s * 0)
                       + (src_alpha << 15)         // B30
#endif
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
#ifdef COMPOSITE_MODE_RGBA
                         + ((s * one_minus_dst_alpha) >> 15)
#endif
                         + ((d * one_minus_src_alpha) >> 15),
                     0, 1<<15);
            }
        }
#ifdef HEAVY_DEBUG
        assert(dst_p[c] <= (1<<15));
        assert(src_p[c] <= (1<<15));
#endif
    }

#ifdef COMPOSITE_MODE_RGBA
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
#endif //HEAVY_DEBUG
#endif //COMPOSITE_MODE_RGBA

#ifdef HEAVY_DEBUG
    assert(dst_p[3] <= (1<<15));
    assert(src_p[3] <= (1<<15));
#endif
}



inline void
#ifdef COMPOSITE_MODE_RGBA
rgba_composite_color_burn_rgba
#else
rgba_composite_color_burn_rgbu
#endif
    (const uint16_t src_p[],
     uint16_t dst_p[],
     const uint16_t opac)
{
    const uint32_t src_alpha30 = (uint32_t)src_p[3]*opac;
    const uint32_t src_alpha = src_alpha30>>15;
    const uint32_t one_minus_src_alpha = (1<<15) - src_alpha;
#ifdef COMPOSITE_MODE_RGBA
    const uint32_t dst_alpha = dst_p[3];
    const uint32_t one_minus_dst_alpha = (1<<15) - dst_alpha;
#else
    const uint32_t dst_alpha = 1<<15;
#endif
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
                dst_p[c] = CLAMP(((d * one_minus_src_alpha) >> 15),
                                 0, (1<<15));
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

#ifdef COMPOSITE_MODE_RGBA
            uint32_t res = (s*one_minus_dst_alpha + d*one_minus_src_alpha)>>15;
            if (dst_alpha > 0) {
                const uint32_t m = (  ((1<<15) - ((d << 15) / dst_alpha))
                                    * src_alpha) / s;
                if (m < (1<<15)) {
                    res += (  ((src_alpha * dst_alpha) >> 15)
                            * ((1<<15) - m) ) >> 15;
                }
            }
#else
            uint32_t res = (d*one_minus_src_alpha)>>15;
            const uint32_t m = (((1<<15) - d)
                                * src_alpha) / s;
            if (m < (1<<15)) {
                res += (  src_alpha
                        * ((1<<15) - m) ) >> 15;
            }
#endif
            dst_p[c] = CLAMP(res, 0, (1<<15));
        }
#ifdef HEAVY_DEBUG
        assert(dst_p[c] <= (1<<15));
        assert(src_p[c] <= (1<<15));
#endif
    }

#ifdef COMPOSITE_MODE_RGBA
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
#endif //HEAVY_DEBUG
#endif //COMPOSITE_MODE_RGBA

#ifdef HEAVY_DEBUG
    assert(dst_p[3] <= (1<<15));
    assert(src_p[3] <= (1<<15));
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


