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


