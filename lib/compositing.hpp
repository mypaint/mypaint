/* This file is part of MyPaint.
 * Copyright (C) 2012 by Andrew Chadwick <a.t.chadwick@gmail.com>
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


/* Specifies how to treat output buffers when compositing into them. */

typedef enum {
    BufferCompOutputRGBA,   // Expect and write full RGBA
    BufferCompOutputRGBX,   // RGB with ignored (opaque) alpha
} BufferCompOutputType;



// Generic buffer compositor. Templated by output type specifier, stateless
// color blend mode functor, and the sizes of the buffers.
//
// Need to template this at the class level to permit partial template
// specialization for more optimized forms. The C++ spec does not permit plain
// functions to be partially specialized.

template <BufferCompOutputType OUTBUFUSAGE,
          unsigned int BUFSIZE,
          typename BLENDFUNC>
class BufferComp
{
  private:
    static BLENDFUNC blendfunc;
  public:
    static inline void composite_src_over
            (const fix15_short_t * const src,
             fix15_short_t * const dst,
             const fix15_short_t opac)
    {
        if (opac == 0) {
            return;
        }
        for (unsigned int i=0; i<BUFSIZE; i+=4) {
            // Leave the backdrop alone if the source is fully transparent
            const fix15_t a_s = fix15_mul(src[i+3], opac);
            if (a_s == 0) {
                continue;
            }
            const fix15_t src_a0 = fix15_mul(opac, src[i]);
            const fix15_t src_a1 = fix15_mul(opac, src[i+1]);
            const fix15_t src_a2 = fix15_mul(opac, src[i+2]);
            const fix15_t a_b = (OUTBUFUSAGE == BufferCompOutputRGBA)
                                ? dst[i+3]
                                : fix15_one ;
            // If the backdrop is empty, the source contributes fully
            if (OUTBUFUSAGE == BufferCompOutputRGBA) {
                if (a_b == 0) {
                    dst[i] = fix15_short_clamp(src_a0);
                    dst[i+1] = fix15_short_clamp(src_a1);
                    dst[i+2] = fix15_short_clamp(src_a2);
                    dst[i+3] = a_s;
                    continue;
                }
            }
            // De-premultiplied version of dst
            fix15_t tmp0 = ((OUTBUFUSAGE == BufferCompOutputRGBA)
                            ? fix15_div(dst[i], a_b)
                            : dst[i]);
            fix15_t tmp1 = ((OUTBUFUSAGE == BufferCompOutputRGBA)
                            ? fix15_div(dst[i+1], a_b)
                            : dst[i+1]);
            fix15_t tmp2 = ((OUTBUFUSAGE == BufferCompOutputRGBA)
                            ? fix15_div(dst[i+2], a_b)
                            : dst[i+2]);
            // Combine using the blend function
            blendfunc(fix15_div(src_a0, a_s),
                      fix15_div(src_a1, a_s),  //de-premult srcs
                      fix15_div(src_a2, a_s),
                      tmp0, tmp1, tmp2);
            // Composite the result using src-over
            const fix15_t asab = (OUTBUFUSAGE == BufferCompOutputRGBA)
                                    ? fix15_mul(a_s, a_b)
                                    : a_s;
            const fix15_t one_minus_a_s = fix15_one - a_s;
            dst[i+0] = fix15_sumprods(one_minus_a_s, dst[i+0],
                                      fix15_short_clamp(tmp0), asab);
            dst[i+1] = fix15_sumprods(one_minus_a_s, dst[i+1],
                                      fix15_short_clamp(tmp1), asab);
            dst[i+2] = fix15_sumprods(one_minus_a_s, dst[i+2],
                                      fix15_short_clamp(tmp2), asab);
            if (OUTBUFUSAGE == BufferCompOutputRGBA) {
                const fix15_t one_minus_a_b = fix15_one - a_b;
                dst[i+0] += fix15_mul(one_minus_a_b, src_a0);
                dst[i+1] += fix15_mul(one_minus_a_b, src_a1);
                dst[i+2] += fix15_mul(one_minus_a_b, src_a2);
                dst[i+3] = fix15_short_clamp(a_s + a_b - asab);
            }
        }
    }
};


#endif //__HAVE_COMPOSITING
