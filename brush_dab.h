#include <gtk/gtk.h>
#include "gtkmysurfaceold.h"
#include "helpers.h"

// The bbox (bounding box) can be NULL, if not, it will be expanded to
// include the surface area which was just painted.
// Returns 0 if nothing was painted.
int draw_brush_dab (GtkMySurfaceOld * s, Rect * bbox,
                    GRand * rng,
                    float x, float y, 
                    float radius, float opaque, float hardness,
                    int color_r, int color_g, int color_b
                    )
     // ; follows after #include
