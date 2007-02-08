#ifndef __MAPPING_H__
#define __MAPPING_H__

// user-defined mappings
// (the curves you can edit in the brush settings)

// private
typedef struct {
  // a set of control points (stepwise linear)
  float xvalues[8];
  float yvalues[8];
  int n;
} ControlPoints;

// only base_value is accessed from outside
typedef struct {
  int inputs;
  float base_value;
  ControlPoints * pointsList; // one for each input
  int inputs_used; // optimization
} Mapping;

Mapping * mapping_new(int inputs);
void mapping_free (Mapping * m);

void mapping_set_n (Mapping * m, int input, int n);
void mapping_set_point (Mapping * m, int input, int index, float x, float y);
float mapping_calculate (Mapping * m, float * inputs);

#endif
