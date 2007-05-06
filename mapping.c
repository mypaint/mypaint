/* This file is part of MyPaint.
 * Copyright (C) 2007 by Martin Renold <martinxyz@gmx.ch>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License.
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY. See the COPYING file for more details.
 */

#include "mapping.h"

#include <glib.h>
//#include <glib-object.h>
//#include <gdk/gdk.h>
//#include <gtk/gtkwidget.h>


Mapping * mapping_new (int inputs)
{
  Mapping * m;
  m = g_new0 (Mapping, 1);
  m->inputs = inputs;
  m->pointsList = g_new0 (ControlPoints, inputs);
  m->inputs_used = 0;
  return m;
}

void mapping_free (Mapping * m)
{
  g_free(m->pointsList);
  g_free(m);
}

void mapping_set_n (Mapping * m, int input, int n)
{
  g_assert (input >= 0 && input < m->inputs);
  g_assert (n >= 0 && n <= 8);
  g_assert (n != 1); // cannot build a linear mapping with only one point
  ControlPoints * p = m->pointsList + input;

  if (n != 0 && p->n == 0) m->inputs_used++;
  if (n == 0 && p->n != 0) m->inputs_used--;
  g_assert(m->inputs_used >= 0);
  g_assert(m->inputs_used <= m->inputs);

  p->n = n;
}

void mapping_set_point (Mapping * m, int input, int index, float x, float y)
{
  g_assert (input >= 0 && input < m->inputs);
  g_assert (index >= 0 && index < 8);
  ControlPoints * p = m->pointsList + input;
  g_assert (index < p->n);

  if (index > 0) {
    g_assert (x > p->xvalues[index-1]);
  }

  p->xvalues[index] = x;
  p->yvalues[index] = y;
}

float mapping_calculate (Mapping * m, float * inputs)
{
  int j;
  float result;
  result = m->base_value;

  // constant mapping (common case)
  if (m->inputs_used == 0) return result;

  for (j=0; j<m->inputs; j++) {
    ControlPoints * p = m->pointsList + j;

    if (p->n) {
      float x, y;
      x = inputs[j];

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

      //g_print ("%d/%d x0=%f,x1=%f\n", i, p->n, x0, x1);

      // linear interpolation
      float m, q;
      m = (y1-y0)/(x1-x0);
      q = y0 - m*x0;
      y = m*x + q;
      result += y;
    }
  }
  return result;
}
