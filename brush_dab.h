#include <gtk/gtk.h>
#include "surface.h"

void draw_brush_dab (Surface * s, GtkWidget * queue_draw_widget, 
                     float x, float y, 
                     float radius, float opaque, float hardness,
                     guchar * color,
                     float saturation_slowdown
                     )
     // ; follows after #include
