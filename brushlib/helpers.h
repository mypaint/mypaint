#ifndef HELPERS_H
#define HELPERS_H

#include <glib.h>

// MAX, MIN, ABS, CLAMP are already available from gmacros.h
#define ROUND(x) ((int) ((x) + 0.5))
#define SIGN(x) ((x)>0?1:(-1))
#define SQR(x) ((x)*(x))

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

float rand_gauss (GRand * rng);

#endif // HELPERS_H
