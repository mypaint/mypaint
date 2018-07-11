/* This file is part of MyPaint.
 * Copyright (C) 2012 by Andrew Chadwick <a.t.chadwick@gmail.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */

// Unsigned scaled integer arithmetic macros, constants, and inline functions.
// Names patterned after libfixmath. 15 bits after the binary point, and either
// one or seventeen bits in front of it.

#ifndef __HAVE_FIX15
#define __HAVE_FIX15

#include <stdint.h>

/* Scaled integer types */

typedef uint32_t fix15_t;   // General arithmetic.
typedef uint16_t fix15_short_t;  // Pixels; for 0 <= n <= fix15_one only.

static const int _fix15_fracbits = 15;
static const fix15_t fix15_one = 1<<_fix15_fracbits;



/* Type conversions */

// TODO: fix15_from_float(), fix15_to_float()
// TODO: fix15_from_uint16(), fix15_to_uint16()   -- 0 to 0xffff



static fix15_short_t
fix15_short_clamp(fix15_t n)
{
    return (n > fix15_one) ? fix15_one : n;
}

/* Basic arithmetic */

// Multiplication of two fix15_t.
static inline fix15_t
fix15_mul (const fix15_t a, const fix15_t b)
{
    return (a * b) >> _fix15_fracbits;
}

// Sum of two fix15_t products.
static inline fix15_t
fix15_sumprods (const fix15_t a1, const fix15_t a2,
                const fix15_t b1, const fix15_t b2)
{
    return ((a1 * a2) + (b1 * b2)) >> _fix15_fracbits;
}

static inline fix15_t
fix15_double (const fix15_t n) {
    return n<<1;
}

static inline fix15_t
fix15_halve (const fix15_t n) {
    return n>>1;
}



// Division of one fix15_t by another.
static inline fix15_t
fix15_div (const fix15_t a, const fix15_t b)
{
    return (a << _fix15_fracbits) / b;
}


/* int15_sqrt:

Square root using the http://en.wikipedia.org/wiki/Babylonian_method . For
inputs in the range 0 to fix15_one, we guarantee

    * Average iterations: 1.977
    * Maximum iterations: 8
    * Inaccuracies: 0

Input is limited to the range [0.0, 1.0], unscaled due to internal rounding
issues.

If we need it to run faster, we can use a bigger lookup table or the binary
approximation method used by libfixmath.
*/

static const uint16_t _int15_sqrt_approx16[] = {
    16383, 23169, 28376, 32767, 36634, 40131, 43346, 46339,
    49151, 51809, 54338, 56754, 59072, 61302, 63453, 65535
};
// Values use a scaling factor of 1/2**16
// in Python: [int(((i/16.0)**0.5)*(1<<16))-1 for i in xrange(1, 17)]


static inline fix15_t
fix15_sqrt (const fix15_t x)
{
#ifdef HEAVY_DEBUG
    assert(x <= fix15_one);
#endif
    if ((x == 0) || (x == fix15_one)) {
        return x;
    }
    // Add one extra bit of precision for working
    // A scaled value of 1.0 would overflow, so inflate type
    uint32_t s = x << 1;
    const int fracbits = _fix15_fracbits + 1;

    // Initial approximation
    uint32_t n = _int15_sqrt_approx16[s>>12];  // s/4096 as index 0..15
    // If we really accurate sqrt() for x >
    // 1.0, we'll need a 64-bit representation since the (s << fracbits) term
    // in the iteration below will overflow a uint32_t. This could use n = x as
    // its approximation for x > 1.0.

    uint32_t n_old = 0;
    // Iterate until converged "closely enough" (ugh).
    for (int i = 0; i < 15; ++i) {
        n_old = n;
        n += (s << fracbits) / n;
        n >>= 1;
        if ((n == n_old)
            || ((n > n_old) && (n-1 == n_old))
            || ((n < n_old) && (n+1 == n_old)))
        {
            break;
        }
    }
    // Lose the extra bit of precision
    return n>>1;
}

#endif // __HAVE_FIX15
