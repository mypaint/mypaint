/* brushlib - The MyPaint Brush Library
 * Copyright (C) 2007-2008 Martin Renold <martinxyz@gmx.ch>
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#ifndef MAPPING_C
#define MAPPING_C

#include <stdlib.h>
#include <assert.h>

#include "mapping.h"

#include "helpers.h"

// user-defined mappings
// (the curves you can edit in the brush settings)

typedef struct {
  // a set of control points (stepwise linear)
  float xvalues[8];
  float yvalues[8];
  int n;
} ControlPoints;

struct _Mapping {
    float base_value; // FIXME: accessed directly from mypaint-brush.c

    int inputs;
    ControlPoints * pointsList; // one for each input
    int inputs_used; // optimization

};


Mapping *
mapping_new(int inputs_)
{
    Mapping *self = (Mapping *)malloc(sizeof(Mapping));

    self->inputs = inputs_;
    self->pointsList = (ControlPoints *)malloc(sizeof(ControlPoints)*self->inputs);
    int i = 0;
    for (i=0; i<self->inputs; i++) self->pointsList[i].n = 0;

    self->inputs_used = 0;
    self->base_value = 0;

    return self;
}

void
mapping_free(Mapping *self)
{
    free(self->pointsList);
    free(self);
}

float mapping_get_base_value(Mapping *self)
{
    return self->base_value;
}

void mapping_set_base_value(Mapping *self, float value)
{
    self->base_value = value;
}

void mapping_set_n (Mapping * self, int input, int n)
{
    assert (input >= 0 && input < self->inputs);
    assert (n >= 0 && n <= 8);
    assert (n != 1); // cannot build a linear mapping with only one point
    ControlPoints * p = self->pointsList + input;

    if (n != 0 && p->n == 0) self->inputs_used++;
    if (n == 0 && p->n != 0) self->inputs_used--;
    assert(self->inputs_used >= 0);
    assert(self->inputs_used <= self->inputs);

    p->n = n;
}


int mapping_get_n (Mapping * self, int input)
{
    assert (input >= 0 && input < self->inputs);
    ControlPoints * p = self->pointsList + input;
    return p->n;
}

void mapping_set_point (Mapping * self, int input, int index, float x, float y)
{
    assert (input >= 0 && input < self->inputs);
    assert (index >= 0 && index < 8);
    ControlPoints * p = self->pointsList + input;
    assert (index < p->n);

    if (index > 0) {
      assert (x >= p->xvalues[index-1]);
    }

    p->xvalues[index] = x;
    p->yvalues[index] = y;
}

void mapping_get_point (Mapping * self, int input, int index, float *x, float *y)
{
    assert (input >= 0 && input < self->inputs);
    assert (index >= 0 && index < 8);
    ControlPoints * p = self->pointsList + input;
    assert (index < p->n);

    *x = p->xvalues[index];
    *y = p->yvalues[index];
}

gboolean mapping_is_constant(Mapping * self)
{
    return self->inputs_used == 0;
}

int
mapping_get_inputs_used_n(Mapping *self)
{
    return self->inputs_used;
}

float mapping_calculate (Mapping * self, float * data)
{
    int j;
    float result;
    result = self->base_value;

    // constant mapping (common case)
    if (self->inputs_used == 0) return result;

    for (j=0; j<self->inputs; j++) {
      ControlPoints * p = self->pointsList + j;

      if (p->n) {
        float x, y;
        x = data[j];

        // find the segment with the slope that we need to use
        float x0, y0, x1, y1;
        x0 = p->xvalues[0];
        y0 = p->yvalues[0];
        x1 = p->xvalues[1];
        y1 = p->yvalues[1];

        int i;
        for (i=2; i<p->n && x>x1; i++) {
          x0 = x1;
          y0 = y1;
          x1 = p->xvalues[i];
          y1 = p->yvalues[i];
        }

        if (x0 == x1) {
          y = y0;
        } else {
          // linear interpolation
          y = (y1*(x - x0) + y0*(x1 - x)) / (x1 - x0);
        }

        result += y;
      }
    }
    return result;
}

// used in python for the global pressure mapping
float mapping_calculate_single_input (Mapping * self, float input)
{
    assert(self->inputs == 1);
    return mapping_calculate(self, &input);
}
#endif //MAPPING_C
