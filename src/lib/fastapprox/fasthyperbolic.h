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

#ifndef __FAST_HYPERBOLIC_H_
#define __FAST_HYPERBOLIC_H_

#include <stdint.h>
#include "sse.h"
#include "fastexp.h"

static inline float
fastsinh (float p)
{
  return 0.5f * (fastexp (p) - fastexp (-p));
}

static inline float
fastersinh (float p)
{
  return 0.5f * (fasterexp (p) - fasterexp (-p));
}

static inline float
fastcosh (float p)
{
  return 0.5f * (fastexp (p) + fastexp (-p));
}

static inline float
fastercosh (float p)
{
  return 0.5f * (fasterexp (p) + fasterexp (-p));
}

static inline float
fasttanh (float p)
{
  return -1.0f + 2.0f / (1.0f + fastexp (-2.0f * p));
}

static inline float
fastertanh (float p)
{
  return -1.0f + 2.0f / (1.0f + fasterexp (-2.0f * p));
}

#ifdef __SSE2__

static inline v4sf
vfastsinh (const v4sf p)
{
  const v4sf c_0_5 = v4sfl (0.5f);

  return c_0_5 * (vfastexp (p) - vfastexp (-p));
}

static inline v4sf
vfastersinh (const v4sf p)
{
  const v4sf c_0_5 = v4sfl (0.5f);

  return c_0_5 * (vfasterexp (p) - vfasterexp (-p));
}

static inline v4sf
vfastcosh (const v4sf p)
{
  const v4sf c_0_5 = v4sfl (0.5f);

  return c_0_5 * (vfastexp (p) + vfastexp (-p));
}

static inline v4sf
vfastercosh (const v4sf p)
{
  const v4sf c_0_5 = v4sfl (0.5f);

  return c_0_5 * (vfasterexp (p) + vfasterexp (-p));
}

static inline v4sf
vfasttanh (const v4sf p)
{
  const v4sf c_1 = v4sfl (1.0f);
  const v4sf c_2 = v4sfl (2.0f);

  return -c_1 + c_2 / (c_1 + vfastexp (-c_2 * p));
}

static inline v4sf
vfastertanh (const v4sf p)
{
  const v4sf c_1 = v4sfl (1.0f);
  const v4sf c_2 = v4sfl (2.0f);

  return -c_1 + c_2 / (c_1 + vfasterexp (-c_2 * p));
}

#endif //__SSE2__

#endif // __FAST_HYPERBOLIC_H_
