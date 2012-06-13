
#include "mypaint-brush-settings.h"

#include <libintl.h>
#include <string.h>
#include <assert.h>

#define GETTEXT_PACKAGE "libmypaint"

#define N_(String) (String)
#define  _(String) gettext (String)

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
    return dgettext(self->name, GETTEXT_PACKAGE);
}

const gchar *
mypaint_brush_setting_info_get_tooltip(const MyPaintBrushSettingInfo *self)
{
    return dgettext(self->tooltip, GETTEXT_PACKAGE);
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
    return -1;
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
    return -1;
}
