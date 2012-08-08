#ifndef HELPERS_H
#define HELPERS_H

#include "rng-double.h"

#define MAX(a, b)  (((a) > (b)) ? (a) : (b))
#define MIN(a, b)  (((a) < (b)) ? (a) : (b))
#define ROUND(x) ((int) ((x) + 0.5))
#define SIGN(x) ((x)>0?1:(-1))
#define SQR(x) ((x)*(x))
#define CLAMP(x, low, high)  (((x) > (high)) ? (high) : (((x) < (low)) ? (low) : (x)))
#define MAX3(a, b, c) ((a)>(b)?MAX((a),(c)):MAX((b),(c)))
#define MIN3(a, b, c) ((a)<(b)?MIN((a),(c)):MIN((b),(c)))

void
hsl_to_rgb_float (float *h_, float *s_, float *l_);
void
rgb_to_hsl_float (float *r_, float *g_, float *b_);

void
hsv_to_rgb_float (float *h_, float *s_, float *v_);

void
rgb_to_hsv_float (float *r_ /*h*/, float *g_ /*s*/, float *b_ /*v*/);

float rand_gauss (RngDouble * rng);

#endif // HELPERS_H
