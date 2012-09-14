#ifndef MYPAINTUTILSSTROKEPLAYER_H
#define MYPAINTUTILSSTROKEPLAYER_H

#include <mypaint-brush.h>
#include <mypaint-surface.h>

typedef struct _MyPaintUtilsStrokePlayer MyPaintUtilsStrokePlayer;

MyPaintUtilsStrokePlayer *
mypaint_utils_stroke_player_new();

void
mypaint_utils_stroke_player_free(MyPaintUtilsStrokePlayer *self);

void
mypaint_utils_stroke_player_set_brush(MyPaintUtilsStrokePlayer *self, MyPaintBrush *brush);

void
mypaint_utils_stroke_player_set_surface(MyPaintUtilsStrokePlayer *self, MyPaintSurface *surface);

void
mypaint_utils_stroke_player_set_source_data(MyPaintUtilsStrokePlayer *self, const char *data);

gboolean
mypaint_utils_stroke_player_iterate(MyPaintUtilsStrokePlayer *self);

void
mypaint_utils_stroke_player_reset(MyPaintUtilsStrokePlayer *self);

void
mypaint_utils_stroke_player_run_sync(MyPaintUtilsStrokePlayer *self);

#endif // MYPAINTUTILSSTROKEPLAYER_H
