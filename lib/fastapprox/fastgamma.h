/*=====================================================================*
 *                   Copyright (C) 2011 Paul Mineiro                   *
 * All rights reserved.                                                *
 *                                                                     *
 * Redistribution and use in source and binary forms, with             *
 * or without modification, are permitted provided that the            *
 * following conditions are met:                                       *
 *                                                                     *
 *     * Redistributions of source code must retain the                *
 *     above copyright notice, this list of conditions and             *
 *     the following disclaimer.                                       *
 *                                                                     *
 *     * Redistributions in binary form must reproduce the             *
 *     above copyright notice, this list of conditions and             *
 *     the following disclaimer in the documentation and/or            *
 *     other materials provided with the distribution.                 *
 *                                                                     *
 *     * Neither the name of Paul Mineiro nor the names                *
 *     of other contributors may be used to endorse or promote         *
 *     products derived from this software without specific            *
 *     prior written permission.                                       *
 *                                                                     *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND              *
 * CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,         *
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES               *
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE             *
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER               *
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,                 *
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES            *
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE           *
 * GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR                *
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF          *
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT           *
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY              *
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE             *
 * POSSIBILITY OF SUCH DAMAGE.                                         *
 *                                                                     *
 * Contact: Paul Mineiro <paul@mineiro.com>                            *
 *=====================================================================*/

#ifndef __FAST_GAMMA_H_
#define __FAST_GAMMA_H_

#include <stdint.h>
#include "sse.h"
#include "fastlog.h"

/* gamma/digamma functions only work for positive inputs */

static inline float
fastlgamma (float x)
{
  float logterm = fastlog (x * (1.0f + x) * (2.0f + x));
  float xp3 = 3.0f + x;

  return - 2.081061466f 
         - x 
         + 0.0833333f / xp3 
         - logterm 
         + (2.5f + x) * fastlog (xp3);
}

static inline float
fasterlgamma (float x)
{
  return - 0.0810614667f 
         - x
         - fasterlog (x)
         + (0.5f + x) * fasterlog (1.0f + x);
}

static inline float
fastdigamma (float x)
{
  float twopx = 2.0f + x;
  float logterm = fastlog (twopx);

  return (-48.0f + x * (-157.0f + x * (-127.0f - 30.0f * x))) /
         (12.0f * x * (1.0f + x) * twopx * twopx)
         + logterm;
}

static inline float
fasterdigamma (float x)
{
  float onepx = 1.0f + x;

  return -1.0f / x - 1.0f / (2 * onepx) + fasterlog (onepx);
}

#ifdef __SSE2__

static inline v4sf
vfastlgamma (v4sf x)
{
  const v4sf c_1_0 = v4sfl (1.0f);
  const v4sf c_2_0 = v4sfl (2.0f);
  const v4sf c_3_0 = v4sfl (3.0f);
  const v4sf c_2_081061466 = v4sfl (2.081061466f);
  const v4sf c_0_0833333 = v4sfl (0.0833333f);
  const v4sf c_2_5 = v4sfl (2.5f);

  v4sf logterm = vfastlog (x * (c_1_0 + x) * (c_2_0 + x));
  v4sf xp3 = c_3_0 + x;

  return - c_2_081061466
         - x 
         + c_0_0833333 / xp3 
         - logterm 
         + (c_2_5 + x) * vfastlog (xp3);
}

static inline v4sf
vfasterlgamma (v4sf x)
{
  const v4sf c_0_0810614667 = v4sfl (0.0810614667f);
  const v4sf c_0_5 = v4sfl (0.5f);
  const v4sf c_1 = v4sfl (1.0f);

  return - c_0_0810614667
         - x
         - vfasterlog (x)
         + (c_0_5 + x) * vfasterlog (c_1 + x);
}

static inline v4sf
vfastdigamma (v4sf x)
{
  v4sf twopx = v4sfl (2.0f) + x;
  v4sf logterm = vfastlog (twopx);

  return (v4sfl (-48.0f) + x * (v4sfl (-157.0f) + x * (v4sfl (-127.0f) - v4sfl (30.0f) * x))) /
         (v4sfl (12.0f) * x * (v4sfl (1.0f) + x) * twopx * twopx)
         + logterm;
}

static inline v4sf
vfasterdigamma (v4sf x)
{
  const v4sf c_1_0 = v4sfl (1.0f);
  const v4sf c_2_0 = v4sfl (2.0f);
  v4sf onepx = c_1_0 + x;

  return -c_1_0 / x - c_1_0 / (c_2_0 * onepx) + vfasterlog (onepx);
}

#endif //__SSE2__

#endif // __FAST_GAMMA_H_
