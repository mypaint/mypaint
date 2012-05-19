#ifndef MAPPING_H
#define MAPPING_H

typedef struct _Mapping Mapping;

Mapping * mapping_new(int inputs_);
void mapping_free(Mapping *self);
float mapping_get_base_value(Mapping *self);
void mapping_set_base_value(Mapping *self, float value);
void mapping_set_n (Mapping * self, int input, int n);
void mapping_set_point (Mapping * self, int input, int index, float x, float y);
bool mapping_is_constant(Mapping * self);
float mapping_calculate (Mapping * self, float * data);
float mapping_calculate_single_input (Mapping * self, float input);

#endif // MAPPING_H
