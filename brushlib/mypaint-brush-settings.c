

#include "mypaint-brush-settings.h"

// FIXME: should be "libmypaint"
#define GETTEXT_PACKAGE "mypaint"
#include <glib/gi18n-lib.h>

#include "brushsettings-gen.h"


const MyPaintBrushSettingInfo *
maliit_brush_setting_info(MyPaintBrushSetting id)
{
    g_return_val_if_fail(id < MYPAINT_BRUSH_SETTINGS_COUNT, NULL);

    return &settings_info_array[id];
}


const MyPaintBrushInputInfo *
maliit_brush_input_info(MyPaintBrushInput id)
{
    g_return_val_if_fail(id < MYPAINT_BRUSH_INPUTS_COUNT, NULL);

    return &inputs_info_array[id];
}
