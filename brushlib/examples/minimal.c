#include "libmypaint.c"
#include "mypaint-brush.h"
#include "mypaint-fixed-tiled-surface.h"

#include "utils.h" /* Not public API, just used for write_ppm to demonstrate */

void
stroke_to(MyPaintBrush *brush, MyPaintSurface *surf, float x, float y)
{
    float pressure = 1.0, ytilt = 0.0, xtilt = 0.0, dtime = 1.0/10;
    mypaint_brush_stroke_to(brush, surf, x, y, pressure, xtilt, ytilt, dtime);
}

int
main(int argc, char argv[]) {

    MyPaintBrush *brush = mypaint_brush_new();
    MyPaintFixedTiledSurface *surface = mypaint_fixed_tiled_surface_new(500, 500);

    mypaint_brush_from_defaults(brush);
    mypaint_brush_set_base_value(brush, MYPAINT_BRUSH_SETTING_COLOR_H, 0.0);
    mypaint_brush_set_base_value(brush, MYPAINT_BRUSH_SETTING_COLOR_S, 1.0);
    mypaint_brush_set_base_value(brush, MYPAINT_BRUSH_SETTING_COLOR_V, 1.0);

    /* Draw a rectangle on surface with brush */
    mypaint_surface_begin_atomic((MyPaintSurface *)surface);
    stroke_to(brush, (MyPaintSurface *)surface, 0.0, 0.0);
    stroke_to(brush, (MyPaintSurface *)surface, 200.0, 0.0);
    stroke_to(brush, (MyPaintSurface *)surface, 200.0, 200.0);
    stroke_to(brush, (MyPaintSurface *)surface, 0.0, 200.0);
    stroke_to(brush, (MyPaintSurface *)surface, 0.0, 0.0);
    mypaint_surface_end_atomic((MyPaintSurface *)surface);

#if 0
    // FIXME: write_ppm is currently broken
    fprintf(stdout, "Writing output\n");
    write_ppm(surface, "output.ppm");
#endif

    mypaint_brush_unref(brush);
    mypaint_surface_unref((MyPaintSurface *)surface);
}
