#ifndef MAPPING_H
#define MAPPING_H

typedef struct _Mapping Mapping;

#include <mypaint-glib-compat.h>

G_BEGIN_DECLS

Mapping * mapping_new(int inputs_);
void mapping_free(Mapping *self);
float mapping_get_base_value(Mapping *self);
void mapping_set_base_value(Mapping *self, float value);
void mapping_set_n (Mapping * self, int input, int n);
int mapping_get_n (Mapping * self, int input);
void mapping_set_point (Mapping * self, int input, int index, float x, float y);
void mapping_get_point (Mapping * self, int input, int index, float *x, float *y);
gboolean mapping_is_constant(Mapping * self);
int mapping_get_inputs_used_n(Mapping *self);
float mapping_calculate (Mapping * self, float * data);
float mapping_calculate_single_input (Mapping * self, float input);


G_END_DECLS

#endif // MAPPING_H
