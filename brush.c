#include <stdio.h>
#include <string.h>
#include <glib.h>
#include <math.h>
#include "brush.h"
#include "helpers.h"
#include "brush_dab.h" 
;  // ; needed

#include "brush_settings.inc"

Brush * brush_create ()
{
  int i;
  Brush * b;
  b = g_new0 (Brush, 1);
  b->queue_draw_widget = NULL;
  for (i=0; brush_setting_infos[i].cname; i++) {
    brush_set_setting (b, i, brush_setting_infos[i].default_value);
  }
  return b;
}

Brush * brush_create_copy (Brush * old_b)
{
  Brush * b;
  b = g_new0 (Brush, 1);
  memcpy (b, old_b, sizeof(Brush));
  return b;
}

void brush_free (Brush * b)
{
  g_free (b);
}

void brush_mutate (Brush * b)
{
#if 0
  int i;
  for (i=0; i<F_WEIGHTS; i++) {
    /*
    if (g_random_int_range(0, 10) == 0) {
      b->variations[i] *= g_random_double_range(0.5, 1.0/0.5);
      }*/
    b->weights[i] += g_random_double_range(-b->variations[i], b->variations[i]);
  }
#endif
}


