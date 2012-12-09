#ifndef MYPAINTUTILSSTROKEPLAYER_H
#define MYPAINTUTILSSTROKEPLAYER_H

/* brushlib - The MyPaint Brush Library
 * Copyright (C) 2012 Jon Nordby <jononor@gmail.com>
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

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

void
mypaint_utils_stroke_player_set_transactions_on_stroke_to(MyPaintUtilsStrokePlayer *self, gboolean value);

void
mypaint_utils_stroke_player_set_scale(MyPaintUtilsStrokePlayer *self, float scale);

#endif // MYPAINTUTILSSTROKEPLAYER_H
