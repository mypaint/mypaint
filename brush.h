#ifndef __brush_h__
#define __brush_h__

#include "surface.h"


extern BrushSettingInfo brush_setting_infos[];

extern Brush * brush_create ();
extern Brush * brush_create_copy (Brush * b);
extern void brush_free (Brush * b);
extern void brush_reset (Brush * b);
extern void brush_stroke_to (Brush * b, Surface * s, float x, float y, float pressure, float time);

extern void brush_set_setting (Brush * b, int id, float value);
extern float brush_get_setting (Brush * b, int id);

extern void brush_mutate (Brush * b);

#endif
