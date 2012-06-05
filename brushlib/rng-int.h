#ifndef RNG_INT_H
#define RNG_INT_H

#include <mypaint-glib-compat.h>

G_BEGIN_DECLS

typedef struct _RngInt RngInt;

RngInt* rng_int_new(long seed);
void rng_int_free(RngInt *self);

void rng_int_set_seed(RngInt *self, long seed);
long rng_int_next(RngInt* self);
void rng_int_get_array(RngInt *self, long aa[],int n);

G_END_DECLS

#endif // RNG_INT_H
