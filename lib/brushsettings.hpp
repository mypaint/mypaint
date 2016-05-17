#ifndef BRUSHSETTINGS_HPP
#define BRUSHSETTINGS_HPP

// Reflect libmypaint brush settings and inputs into Python code

#include "Python.h"

// Get information about brush settings known to libmypaint.
// Return value is a list of dicts reflecting all "MyPaintBrushSettingInfo"s.
// Only difference: tooltips and names in the result are pre-translated.

PyObject *
get_libmypaint_brush_settings();

// Get information about brush inputs known to libmypaint.
// Return value is a list of dicts reflecting all "MyPaintBrushInputInfo"s.
// Only difference: tooltips and names in the result are pre-translated.

PyObject *
get_libmypaint_brush_inputs();


#endif // BRUSHSETTINGS.HPP
