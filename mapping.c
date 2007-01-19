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

void mapping_set (Mapping * m, int input, int index, float value)
{
  g_assert (input >= 0 && input < m->inputs);
  g_assert (index >= 0 && index < 8);
  ControlPoints * p = m->pointsList + input;

  if (index == 0) {
    float old = p->xvalues[0];
    if (value != 0 && old == 0) m->inputs_used++;
    if (value == 0 && old != 0) m->inputs_used--;
    g_assert(m->inputs_used >= 0);
    g_assert(m->inputs_used <= m->inputs);
  }

  if (index % 2 == 0) {
    p->xvalues[index/2] = value;
  } else {
    p->yvalues[index/2] = value;
  }
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
    if (p->xvalues[0]) {
      float x, y;
      x = inputs[j];
      //if (i == 2 && j == 3) g_print("x = %f ", x);
      int p0;
      float x0, y0, x1, y1;
      // decide what region to use
      p0 = -1; // left point of the linear region (-1 is the implicit x=0,y=0 point)
      while (p0+1 < 4 // not in the last region already
             && x > p->xvalues[p0+1] // x position is further right than the current region
             && p->xvalues[p0+2] > 0 // next enpoint does exists (points with x=0 are disabled)
             ) p0++;
      x0 = (p0 == -1) ? 0 : p->xvalues[p0];
      y0 = (p0 == -1) ? 0 : p->yvalues[p0];
      x1 = p->xvalues[p0+1];
      y1 = p->yvalues[p0+1];
      // linear interpolation
      float m, q;
      m = (y1-y0)/(x1-x0);
      q = y0 - m*x0;
      y = m*x + q;
      result += y;
      //if (i == 2 && j == 3) g_print("y = %f (p0=%d, %f %f %f %f)\n", y, p0, x0, y0, x1, y1);
    }
  }
  return result;
}
