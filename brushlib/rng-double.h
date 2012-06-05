#ifndef RNGDOUBLE_H
#define RNGDOUBLE_H

#include <mypaint-glib-compat.h>

G_BEGIN_DECLS

typedef struct _RngDouble RngDouble;

RngDouble* rng_double_new(long seed);
void rng_double_free(RngDouble *self);

void rng_double_set_seed(RngDouble *self, long seed);
double rng_double_next(RngDouble* self);
void rng_double_get_array(RngDouble *self, double aa[], int n);

G_END_DECLS

#endif // RNGDOUBLE_H
