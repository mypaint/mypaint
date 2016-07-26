/* brushlib - The MyPaint Brush Library
 * Copyright (C) 2007-2011 Martin Renold <martinxyz@gmx.ch>
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

/** Brush:
 *
 * C++ wrapper around MyPaintBrush. */
class Brush {

public:
  Brush() {
      c_brush = mypaint_brush_new();
  }

  ~Brush() {
      mypaint_brush_unref(c_brush);
  }

  void reset()
  {
      mypaint_brush_reset(c_brush);
  }

  void new_stroke()
  {
      mypaint_brush_new_stroke(c_brush);
  }

  void set_base_value (int id, float value) {
      mypaint_brush_set_base_value(c_brush, (MyPaintBrushSetting)id, value);
  }

  void set_mapping_n (int id, int input, int n) {
      mypaint_brush_set_mapping_n(c_brush, (MyPaintBrushSetting)id, (MyPaintBrushInput)input, n);
  }

  void set_mapping_point (int id, int input, int index, float x, float y) {
      mypaint_brush_set_mapping_point(c_brush, (MyPaintBrushSetting)id, (MyPaintBrushInput)input, index, x, y);
  }

  float get_state (int i)
  {
      return mypaint_brush_get_state(c_brush, (MyPaintBrushState)i);
  }

  void set_state (int i, float value)
  {
      mypaint_brush_set_state(c_brush, (MyPaintBrushState)i, value);
  }

  bool stroke_to (Surface * surface, float x, float y, float pressure, float xtilt, float ytilt, double dtime, float viewzoom, float viewrotation, float barrel_rotation)
  {
      MyPaintSurface *c_surface = surface->get_surface_interface();
      bool retval = mypaint_brush_stroke_to(c_brush, c_surface, x, y, pressure, xtilt, ytilt, dtime, viewzoom, viewrotation, barrel_rotation);
      return retval;
  }

  double get_total_stroke_painting_time()
  {
      return mypaint_brush_get_total_stroke_painting_time(c_brush);
  }

  void set_print_inputs(bool enabled)
  {
      mypaint_brush_set_print_inputs(c_brush, enabled);
  }

private:
  MyPaintBrush *c_brush;
};
