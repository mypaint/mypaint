#ifndef __MAPPING_H__
#define __MAPPING_H__

// user-defined mappings
// (the curves you can edit in the brush settings)

// private
typedef struct {
  // a set of control points (stepwise linear)
  float xvalues[4]; // >= 0 because all inputs are >= 0
  float yvalues[4]; // range: -oo  .. +oo (added to base_value)
  // xvalues can be zero to indicate that this point is not used.
  // the first point (0, 0) is implicit, would have index -1
} ControlPoints;

// only base_value is accessed from outside
typedef struct {
  int inputs;
  float base_value;
  ControlPoints * pointsList; // one for each input
} Mapping;

Mapping * mapping_new(int inputs);
void mapping_free (Mapping * m);

void mapping_set (Mapping * m, int input, int index, float value);
float mapping_calculate (Mapping * m, float * inputs);

#endif
