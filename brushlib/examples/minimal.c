#include "libmypaint.c"
#include "mypaint-brush.h"
#include "mypaint-fixed-tiled-surface.h"

#include "utils.h" /* Not public API, just used for write_ppm to demonstrate */

void
stroke_to(MyPaintBrush *brush, MyPaintSurface *surf, float x, float y)
{
    float pressure = 1.0, ytilt = 0.0, xtilt = 0.0, dtime = 100.0;
    mypaint_brush_stroke_to(brush, surf, 0.0, 0.0, pressure, xtilt, ytilt, dtime);
}

int
main(int argc, char argv[]) {

    MyPaintBrush *brush = mypaint_brush_new();
    MyPaintFixedTiledSurface *surface = mypaint_fixed_tiled_surface_new(500, 500);

    /* Draw a rectangle on surface with brush */
    stroke_to(brush, (MyPaintSurface *)surface, 0.0, 0.0);
    stroke_to(brush, (MyPaintSurface *)surface, 200.0, 0.0);
    stroke_to(brush, (MyPaintSurface *)surface, 200.0, 200.0);
    stroke_to(brush, (MyPaintSurface *)surface, 0.0, 200.0);
    stroke_to(brush, (MyPaintSurface *)surface, 0.0, 0.0);

    fprintf(stdout, "Writing output\n");
    write_ppm(surface, "output.ppm");

    mypaint_brush_unref(brush);
    mypaint_surface_unref((MyPaintSurface *)surface);
}
