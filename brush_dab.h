#include <gtk/gtk.h>
#include "surface.h"
#include "helpers.h"

void draw_brush_dab (Surface * s, Rect * bbox,
                     float x, float y, 
                     float radius, float opaque, float hardness,
                     guchar * color,
                     float saturation_slowdown
                     )
     // ; follows after #include
