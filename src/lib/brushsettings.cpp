
// Reflect brush setting and input information from libmypaint into Python.

// This probably isn't the best way of doing this,
// given that we have SWIG to hand.
// It does ensure that the human-readable fields are translated, however.
// I am OK with this.

#include "brushsettings.hpp"

#include <mypaint-brush-settings.h>


PyObject *
get_libmypaint_brush_settings()
{
    PyObject* result = PyList_New(0);
    if (! result) {
        PyErr_SetString(PyExc_MemoryError, "Unable to create result list");
        return NULL;
    }
    for (int i = 0; i < MYPAINT_BRUSH_SETTINGS_COUNT; ++i) {
        MyPaintBrushSetting id = (MyPaintBrushSetting) i;
        const MyPaintBrushSettingInfo *inf = mypaint_brush_setting_info(id);
        if (! inf) {
            PyErr_SetString(PyExc_RuntimeError,
                            "Unable to get brush setting from libmypaint");
            break;
        }
        PyObject *inf_dict = Py_BuildValue(
            "{s:s,s:s,s:b,s:f,s:f,s:f,s:s}",
            "cname", inf->cname,
            "name", mypaint_brush_setting_info_get_name(inf),
            "constant", inf->constant,
            "min", inf->min,
            "default", inf->def,
            "max", inf->max,
            "tooltip", mypaint_brush_setting_info_get_tooltip(inf)
        );
        if (! inf_dict) {
            PyErr_SetString(PyExc_MemoryError, "Unable to create item dict");
            break;
        }
        PyList_Append(result, inf_dict);
    }
    return result;
}


PyObject *
get_libmypaint_brush_inputs()
{
    PyObject* result = PyList_New(0);
    if (! result) {
        PyErr_SetString(PyExc_MemoryError, "Unable to create result list");
        return NULL;
    }
    for (int i = 0; i < MYPAINT_BRUSH_INPUTS_COUNT; ++i) {
        MyPaintBrushInput id = (MyPaintBrushInput) i;
        const MyPaintBrushInputInfo *inf = mypaint_brush_input_info(id);
        if (! inf) {
            PyErr_SetString(PyExc_RuntimeError,
                            "Unable to get brush input info from libmypaint");
            break;
        }
        PyObject *inf_dict = Py_BuildValue(
            "{s:s,s:f,s:f,s:f,s:f,s:f,s:s,s:s}",
            "name", inf->cname,
            "hard_min", inf->hard_min,
            "soft_min", inf->soft_min,
            "normal", inf->normal,
            "hard_max", inf->hard_max,
            "soft_max", inf->soft_max,
            "dname", mypaint_brush_input_info_get_name(inf),
            "tooltip", mypaint_brush_input_info_get_tooltip(inf)
        );
        if (! inf_dict) {
            PyErr_SetString(PyExc_MemoryError, "Unable to create item dict");
            break;
        }
        PyList_Append(result, inf_dict);
    }
    return result;
}

