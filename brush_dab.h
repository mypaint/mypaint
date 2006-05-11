#include <gtk/gtk.h>
#include "surface.h"
#include "helpers.h"

// The bbox (bounding box) can be NULL, if not, it will be expanded to
// include the surface area which was just painted.
void draw_brush_dab (Surface * s, Rect * bbox,
                     float x, float y, 
                     float radius, float opaque, float hardness,
                     guchar * color,
                     float saturation_slowdown
                     )
     // ; follows after #include
