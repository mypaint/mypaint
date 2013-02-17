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

#include "mypaint-brush-settings.h"

#include <string.h>
#include <assert.h>

#define GETTEXT_PACKAGE "libmypaint"

#ifdef HAVE_GETTEXT
  #include <libintl.h>
  #define N_(String) (String)
  #define  _(String) gettext (String)
#else
  #define dgettext(a,b) (a)
  #define N_(String) (String)
  #define  _(String) (String)
#endif // HAVE_GETTEXT

#include <float.h>

#include "brushsettings-gen.h"

const MyPaintBrushSettingInfo *
mypaint_brush_setting_info(MyPaintBrushSetting id)
{
    assert(id < MYPAINT_BRUSH_SETTINGS_COUNT);

    return &settings_info_array[id];
}

const gchar *
mypaint_brush_setting_info_get_name(const MyPaintBrushSettingInfo *self)
{
    return dgettext(GETTEXT_PACKAGE, self->name);
}

const gchar *
mypaint_brush_setting_info_get_tooltip(const MyPaintBrushSettingInfo *self)
{
    return dgettext(GETTEXT_PACKAGE, self->tooltip);
}

MyPaintBrushSetting
mypaint_brush_setting_from_cname(const char *cname)
{
    for (int i=0; i<MYPAINT_BRUSH_SETTINGS_COUNT; i++) {
        MyPaintBrushSetting id = (MyPaintBrushSetting)i;
        if (strcmp(mypaint_brush_setting_info(id)->cname, cname) == 0) {
            return id;
        }
    }
    return (MyPaintBrushSetting)-1;
}

const MyPaintBrushInputInfo *
mypaint_brush_input_info(MyPaintBrushInput id)
{
    assert(id < MYPAINT_BRUSH_INPUTS_COUNT);

    return &inputs_info_array[id];
}

const gchar *
mypaint_brush_input_info_get_name(const MyPaintBrushInputInfo *self)
{
    return dgettext(self->name, GETTEXT_PACKAGE);
}

const gchar *
mypaint_brush_input_info_get_tooltip(const MyPaintBrushInputInfo *self)
{
    return dgettext(self->tooltip, GETTEXT_PACKAGE);
}

MyPaintBrushInput
mypaint_brush_input_from_cname(const char *cname)
{
    for (int i=0; i<MYPAINT_BRUSH_INPUTS_COUNT; i++) {
        MyPaintBrushInput id = (MyPaintBrushInput)i;
        if (strcmp(mypaint_brush_input_info(id)->cname, cname) == 0) {
            return id;
        }
    }
    return (MyPaintBrushInput)-1;
}
