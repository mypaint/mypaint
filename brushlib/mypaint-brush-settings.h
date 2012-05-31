#ifndef MYPAINTBRUSHSETTINGS_H
#define MYPAINTBRUSHSETTINGS_H

#include <glib.h>

#include <mypaint-brush-settings-gen.h>

typedef struct {
    const gchar *cname;
    const gchar *name; // FIXME: needs to be accessed through gettext
    gboolean constant;
    float min;
    float def; // default
    float max;
    const gchar *tooltip; // FIXME: needs to be accessed through gettext
} MyPaintBrushSettingInfo;

const MyPaintBrushSettingInfo *
maliit_brush_setting_info(MyPaintBrushSetting id);

const gchar *
maliit_brush_setting_info_get_name(const MyPaintBrushSettingInfo *self);
const gchar *
maliit_brush_setting_info_get_tooltip(const MyPaintBrushSettingInfo *self);

typedef struct {
    const gchar *cname;
    float hard_min;
    float soft_min;
    float normal;
    float soft_max;
    float hard_max;
    const gchar *name; // FIXME: needs to be accessed through gettext
    const gchar *tooltip; // FIXME: needs to be accessed through gettext
} MyPaintBrushInputInfo;

const MyPaintBrushInputInfo *
maliit_brush_input_info(MyPaintBrushInput id);

const gchar *
maliit_brush_input_info_get_name(const MyPaintBrushInputInfo *self);

const gchar *
maliit_brush_input_info_get_tooltip(const MyPaintBrushInputInfo *self);

#endif // MYPAINTBRUSHSETTINGS_H
