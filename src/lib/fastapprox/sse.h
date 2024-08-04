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

#ifndef __SSE_H_
#define __SSE_H_

#ifdef __SSE2__

#include <emmintrin.h>

#ifdef __cplusplus
namespace {
#endif // __cplusplus

typedef __m128 v4sf;
typedef __m128i v4si;

#define v4si_to_v4sf _mm_cvtepi32_ps
#define v4sf_to_v4si _mm_cvttps_epi32

#if _MSC_VER && !__INTEL_COMPILER
  template <class T>
  __forceinline char GetChar(T value, size_t index) { return ((char*)&value)[index]; }

  #define AS_4CHARS(a) \
      GetChar(int32_t(a), 0), GetChar(int32_t(a), 1), \
      GetChar(int32_t(a), 2), GetChar(int32_t(a), 3)

  #define _MM_SETR_EPI32(a0, a1, a2, a3) \
      { AS_4CHARS(a0), AS_4CHARS(a1), AS_4CHARS(a2), AS_4CHARS(a3) }

  #define v4sfl(x) (const v4sf { (x), (x), (x), (x) })
  #define v4sil(x) (const v4si _MM_SETR_EPI32(x, x, x, x))

  __forceinline const v4sf operator+(const v4sf& a, const v4sf& b) { return _mm_add_ps(a,b); }
  __forceinline const v4sf operator-(const v4sf& a, const v4sf& b) { return _mm_sub_ps(a,b); }
  __forceinline const v4sf operator/(const v4sf& a, const v4sf& b) { return _mm_div_ps(a,b); }
  __forceinline const v4sf operator*(const v4sf& a, const v4sf& b) { return _mm_mul_ps(a,b); }

  __forceinline const v4sf operator+(const v4sf& a) { return a; }
  __forceinline const v4sf operator-(const v4sf& a) { return _mm_xor_ps(a, _mm_castsi128_ps(_mm_set1_epi32(0x80000000))); }

  __forceinline const v4sf operator&(const v4sf& a, const v4sf& b) { return _mm_and_ps(a,b); }
  __forceinline const v4sf operator|(const v4sf& a, const v4sf& b) { return _mm_or_ps(a,b); }
  __forceinline const v4sf operator^(const v4sf& a, const v4sf& b) { return _mm_xor_ps(a,b); }

  __forceinline const v4si operator&(const v4si& a, const v4si& b) { return _mm_and_si128(a,b); }
  __forceinline const v4si operator|(const v4si& a, const v4si& b) { return _mm_or_si128(a,b); }
  __forceinline const v4si operator^(const v4si& a, const v4si& b) { return _mm_xor_si128(a,b); }

  __forceinline const v4sf operator+=(v4sf& a, const v4sf& b) { return a = a + b; }
  __forceinline const v4sf operator-=(v4sf& a, const v4sf& b) { return a = a - b; }
  __forceinline const v4sf operator*=(v4sf& a, const v4sf& b) { return a = a * b; }
  __forceinline const v4sf operator/=(v4sf& a, const v4sf& b) { return a = a / b; }

  __forceinline const v4si operator|=(v4si& a, const v4si& b) { return a = a | b; }
  __forceinline const v4si operator&=(v4si& a, const v4si& b) { return a = a & b; }
  __forceinline const v4si operator^=(v4si& a, const v4si& b) { return a = a ^ b; }
#else
  #define v4sfl(x) ((const v4sf) { (x), (x), (x), (x) })
  #define v2dil(x) ((const v4si) { (x), (x) })
  #define v4sil(x) v2dil((((long long) (x)) << 32) | (long long) (x))
#endif

typedef union { v4sf f; float array[4]; } v4sfindexer;
#define v4sf_index(_findx, _findi)      \
  ({                                    \
     v4sfindexer _findvx = { _findx } ; \
     _findvx.array[_findi];             \
  })
typedef union { v4si i; int array[4]; } v4siindexer;
#define v4si_index(_iindx, _iindi)      \
  ({                                    \
     v4siindexer _iindvx = { _iindx } ; \
     _iindvx.array[_iindi];             \
  })

typedef union { v4sf f; v4si i; } v4sfv4sipun;
#if _MSC_VER && !__INTEL_COMPILER
  #define v4sf_fabs(x) _mm_and_ps(x, _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff)))
#else
  #define v4sf_fabs(x)                  \
  ({                                    \
     v4sfv4sipun vx;                    \
     vx.f = x;                          \
     vx.i &= v4sil (0x7FFFFFFF);        \
     vx.f;                              \
  })
#endif

#ifdef __cplusplus
} // end namespace
#endif // __cplusplus

#endif // __SSE2__

#endif // __SSE_H_
