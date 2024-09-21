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

#ifndef __FAST_TRIG_H_
#define __FAST_TRIG_H_

#include <stdint.h>
#include "sse.h"

// http://www.devmaster.net/forums/showthread.php?t=5784
// fast sine variants are for x \in [ -\pi, pi ]
// fast cosine variants are for x \in [ -\pi, pi ]
// fast tangent variants are for x \in [ -\pi / 2, pi / 2 ]
// "full" versions of functions handle the entire range of inputs
// although the range reduction technique used here will be hopelessly
// inaccurate for |x| >> 1000
//
// WARNING: fastsinfull, fastcosfull, and fasttanfull can be slower than
// libc calls on older machines (!) and on newer machines are only 
// slighly faster.  however:
//   * vectorized versions are competitive
//   * faster full versions are competitive

static inline float
fastsin (float x)
{
  static const float fouroverpi = 1.2732395447351627f;
  static const float fouroverpisq = 0.40528473456935109f;
  static const float q = 0.78444488374548933f;
  union { float f; uint32_t i; } p = { 0.20363937680730309f };
  union { float f; uint32_t i; } r = { 0.015124940802184233f };
  union { float f; uint32_t i; } s = { -0.0032225901625579573f };

  union { float f; uint32_t i; } vx = { x };
  uint32_t sign = vx.i & 0x80000000;
  vx.i = vx.i & 0x7FFFFFFF;

  float qpprox = fouroverpi * x - fouroverpisq * x * vx.f;
  float qpproxsq = qpprox * qpprox;

  p.i |= sign;
  r.i |= sign;
  s.i ^= sign;

  return q * qpprox + qpproxsq * (p.f + qpproxsq * (r.f + qpproxsq * s.f));
}

static inline float
fastersin (float x)
{
  static const float fouroverpi = 1.2732395447351627f;
  static const float fouroverpisq = 0.40528473456935109f;
  static const float q = 0.77633023248007499f;
  union { float f; uint32_t i; } p = { 0.22308510060189463f };

  union { float f; uint32_t i; } vx = { x };
  uint32_t sign = vx.i & 0x80000000;
  vx.i &= 0x7FFFFFFF;

  float qpprox = fouroverpi * x - fouroverpisq * x * vx.f;

  p.i |= sign;

  return qpprox * (q + p.f * qpprox);
}

static inline float
fastsinfull (float x)
{
  static const float twopi = 6.2831853071795865f;
  static const float invtwopi = 0.15915494309189534f;

  int k = x * invtwopi;
  float half = (x < 0) ? -0.5f : 0.5f;
  return fastsin ((half + k) * twopi - x);
}

static inline float
fastersinfull (float x)
{
  static const float twopi = 6.2831853071795865f;
  static const float invtwopi = 0.15915494309189534f;

  int k = x * invtwopi;
  float half = (x < 0) ? -0.5f : 0.5f;
  return fastersin ((half + k) * twopi - x);
}

static inline float
fastcos (float x)
{
  static const float halfpi = 1.5707963267948966f;
  static const float halfpiminustwopi = -4.7123889803846899f;
  float offset = (x > halfpi) ? halfpiminustwopi : halfpi;
  return fastsin (x + offset);
}

static inline float
fastercos (float x)
{
  static const float twooverpi = 0.63661977236758134f;
  static const float p = 0.54641335845679634f;

  union { float f; uint32_t i; } vx = { x };
  vx.i &= 0x7FFFFFFF;

  float qpprox = 1.0f - twooverpi * vx.f;

  return qpprox + p * qpprox * (1.0f - qpprox * qpprox);
}

static inline float
fastcosfull (float x)
{
  static const float halfpi = 1.5707963267948966f;
  return fastsinfull (x + halfpi);
}

static inline float
fastercosfull (float x)
{
  static const float halfpi = 1.5707963267948966f;
  return fastersinfull (x + halfpi);
}

static inline float
fasttan (float x)
{
  static const float halfpi = 1.5707963267948966f;
  return fastsin (x) / fastsin (x + halfpi);
}

static inline float
fastertan (float x)
{
  return fastersin (x) / fastercos (x);
}

static inline float
fasttanfull (float x)
{
  static const float twopi = 6.2831853071795865f;
  static const float invtwopi = 0.15915494309189534f;

  int k = x * invtwopi;
  float half = (x < 0) ? -0.5f : 0.5f;
  float xnew = x - (half + k) * twopi;

  return fastsin (xnew) / fastcos (xnew);
}

static inline float
fastertanfull (float x)
{
  static const float twopi = 6.2831853071795865f;
  static const float invtwopi = 0.15915494309189534f;

  int k = x * invtwopi;
  float half = (x < 0) ? -0.5f : 0.5f;
  float xnew = x - (half + k) * twopi;

  return fastersin (xnew) / fastercos (xnew);
}

#ifdef __SSE2__

static inline v4sf
vfastsin (const v4sf x)
{
  const v4sf fouroverpi = v4sfl (1.2732395447351627f);
  const v4sf fouroverpisq = v4sfl (0.40528473456935109f);
  const v4sf q = v4sfl (0.78444488374548933f);
  const v4sf p = v4sfl (0.20363937680730309f);
  const v4sf r = v4sfl (0.015124940802184233f);
  const v4sf s = v4sfl (-0.0032225901625579573f);

  union { v4sf f; v4si i; } vx = { x };
  v4si sign = vx.i & v4sil (0x80000000);
  vx.i &= v4sil (0x7FFFFFFF);

  v4sf qpprox = fouroverpi * x - fouroverpisq * x * vx.f;
  v4sf qpproxsq = qpprox * qpprox;
  union { v4sf f; v4si i; } vy; vy.f = qpproxsq * (p + qpproxsq * (r + qpproxsq * s));
  vy.i ^= sign;

  return q * qpprox + vy.f;
}

static inline v4sf
vfastersin (const v4sf x)
{
  const v4sf fouroverpi = v4sfl (1.2732395447351627f);
  const v4sf fouroverpisq = v4sfl (0.40528473456935109f);
  const v4sf q = v4sfl (0.77633023248007499f);
  const v4sf plit = v4sfl (0.22308510060189463f);
  union { v4sf f; v4si i; } p = { plit };

  union { v4sf f; v4si i; } vx = { x };
  v4si sign = vx.i & v4sil (0x80000000);
  vx.i &= v4sil (0x7FFFFFFF);

  v4sf qpprox = fouroverpi * x - fouroverpisq * x * vx.f;

  p.i |= sign;

  return qpprox * (q + p.f * qpprox);
}

static inline v4sf
vfastsinfull (const v4sf x)
{
  const v4sf twopi = v4sfl (6.2831853071795865f);
  const v4sf invtwopi = v4sfl (0.15915494309189534f);

  v4si k = v4sf_to_v4si (x * invtwopi);

  v4sf ltzero = _mm_cmplt_ps (x, v4sfl (0.0f));
  v4sf half = _mm_or_ps (_mm_and_ps (ltzero, v4sfl (-0.5f)),
                         _mm_andnot_ps (ltzero, v4sfl (0.5f)));

  return vfastsin ((half + v4si_to_v4sf (k)) * twopi - x);
}

static inline v4sf
vfastersinfull (const v4sf x)
{
  const v4sf twopi = v4sfl (6.2831853071795865f);
  const v4sf invtwopi = v4sfl (0.15915494309189534f);

  v4si k = v4sf_to_v4si (x * invtwopi);

  v4sf ltzero = _mm_cmplt_ps (x, v4sfl (0.0f));
  v4sf half = _mm_or_ps (_mm_and_ps (ltzero, v4sfl (-0.5f)),
                         _mm_andnot_ps (ltzero, v4sfl (0.5f)));

  return vfastersin ((half + v4si_to_v4sf (k)) * twopi - x);
}

static inline v4sf
vfastcos (const v4sf x)
{
  const v4sf halfpi = v4sfl (1.5707963267948966f);
  const v4sf halfpiminustwopi = v4sfl (-4.7123889803846899f);
  v4sf lthalfpi = _mm_cmpnlt_ps (x, halfpi);
  v4sf offset = _mm_or_ps (_mm_and_ps (lthalfpi, halfpiminustwopi),
                           _mm_andnot_ps (lthalfpi, halfpi));
  return vfastsin (x + offset);
}

static inline v4sf
vfastercos (v4sf x)
{
  const v4sf twooverpi = v4sfl (0.63661977236758134f);
  const v4sf p = v4sfl (0.54641335845679634);

  v4sf vx = v4sf_fabs (x);
  v4sf qpprox = v4sfl (1.0f) - twooverpi * vx;

  return qpprox + p * qpprox * (v4sfl (1.0f) - qpprox * qpprox);
}

static inline v4sf
vfastcosfull (const v4sf x)
{
  const v4sf halfpi = v4sfl (1.5707963267948966f);
  return vfastsinfull (x + halfpi);
}

static inline v4sf
vfastercosfull (const v4sf x)
{
  const v4sf halfpi = v4sfl (1.5707963267948966f);
  return vfastersinfull (x + halfpi);
}

static inline v4sf
vfasttan (const v4sf x)
{
  const v4sf halfpi = v4sfl (1.5707963267948966f);
  return vfastsin (x) / vfastsin (x + halfpi);
}

static inline v4sf
vfastertan (const v4sf x)
{
  return vfastersin (x) / vfastercos (x);
}

static inline v4sf
vfasttanfull (const v4sf x)
{
  const v4sf twopi = v4sfl (6.2831853071795865f);
  const v4sf invtwopi = v4sfl (0.15915494309189534f);

  v4si k = v4sf_to_v4si (x * invtwopi);

  v4sf ltzero = _mm_cmplt_ps (x, v4sfl (0.0f));
  v4sf half = _mm_or_ps (_mm_and_ps (ltzero, v4sfl (-0.5f)),
                         _mm_andnot_ps (ltzero, v4sfl (0.5f)));
  v4sf xnew = x - (half + v4si_to_v4sf (k)) * twopi;

  return vfastsin (xnew) / vfastcos (xnew);
}

static inline v4sf
vfastertanfull (const v4sf x)
{
  const v4sf twopi = v4sfl (6.2831853071795865f);
  const v4sf invtwopi = v4sfl (0.15915494309189534f);

  v4si k = v4sf_to_v4si (x * invtwopi);

  v4sf ltzero = _mm_cmplt_ps (x, v4sfl (0.0f));
  v4sf half = _mm_or_ps (_mm_and_ps (ltzero, v4sfl (-0.5f)),
                         _mm_andnot_ps (ltzero, v4sfl (0.5f)));
  v4sf xnew = x - (half + v4si_to_v4sf (k)) * twopi;

  return vfastersin (xnew) / vfastercos (xnew);
}

#endif //__SSE2__

#endif // __FAST_TRIG_H_
