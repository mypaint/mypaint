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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "mypaint-brush.h"

#include "mypaint-brush-settings.h"
#include "mapping.h"
#include "helpers.h"
#include "rng-double.h"

#ifdef HAVE_JSON_C
// Allow the C99 define from json.h
#undef TRUE
#undef FALSE
#include <json.h>
#endif // HAVE_JSON_C

#ifdef _MSC_VER
  #include <float.h>
  static inline int    isfinite(double x) { return _finite(x); }
  static inline float  roundf  (float  x) { return x >= 0.0f ? floorf(x + 0.5f) : ceilf(x - 0.5f); }
#endif

#define M_PI 3.14159265358979323846

#define ACTUAL_RADIUS_MIN 0.2
#define ACTUAL_RADIUS_MAX 1000 // safety guard against radius like 1e20 and against rendering overload with unexpected brush dynamics

/* The Brush class stores two things:
   b) settings: constant during a stroke (eg. size, spacing, dynamics, color selected by the user)
   a) states: modified during a stroke (eg. speed, smudge colors, time/distance to next dab, position filter states)

   FIXME: Actually those are two orthogonal things. Should separate them:
          a) brush settings class that is saved/loaded/selected  (without states)
          b) brush core class to draw the dabs (using an instance of the above)

   In python, there are two kinds of instances from this: a "global
   brush" which does the cursor tracking, and the "brushlist" where
   the states are ignored. When a brush is selected, its settings are
   copied into the global one, leaving the state intact.
 */


/**
  * MyPaintBrush:
  *
  * The MyPaint brush engine class.
  */
struct _MyPaintBrush {

    gboolean print_inputs; // debug menu
    // for stroke splitting (undo/redo)
    double stroke_total_painting_time;
    double stroke_current_idling_time;

    // the states (get_state, set_state, reset) that change during a stroke
    float states[MYPAINT_BRUSH_STATES_COUNT];
    RngDouble * rng;

    // Those mappings describe how to calculate the current value for each setting.
    // Most of settings will be constant (eg. only their base_value is used).
    Mapping * settings[MYPAINT_BRUSH_SETTINGS_COUNT];

    // the current value of all settings (calculated using the current state)
    float settings_value[MYPAINT_BRUSH_SETTINGS_COUNT];

    // see also brushsettings.py

    // cached calculation results
    float speed_mapping_gamma[2];
    float speed_mapping_m[2];
    float speed_mapping_q[2];

    gboolean reset_requested;
#ifdef HAVE_JSON_C
    json_object *brush_json;
#endif
    int refcount;
};


void settings_base_values_have_changed (MyPaintBrush *self);

#include "glib/mypaint-brush.c"

/**
  * mypaint_brush_new:
  *
  * Create a new MyPaint brush engine instance.
  * Initial reference count is 1. Release references using mypaint_brush_unref()
  */
MyPaintBrush *
mypaint_brush_new()
{
    MyPaintBrush *self = (MyPaintBrush *)malloc(sizeof(MyPaintBrush));

    self->refcount = 1;
    int i=0;
    for (i=0; i<MYPAINT_BRUSH_SETTINGS_COUNT; i++) {
      self->settings[i] = mapping_new(MYPAINT_BRUSH_INPUTS_COUNT);
    }
    self->rng = rng_double_new(1000);
    self->print_inputs = FALSE;

    for (i=0; i<MYPAINT_BRUSH_STATES_COUNT; i++) {
      self->states[i] = 0;
    }
    mypaint_brush_new_stroke(self);

    settings_base_values_have_changed(self);

    self->reset_requested = TRUE;

#ifdef HAVE_JSON_C
    self->brush_json = json_object_new_object();
#endif

    return self;
}

void
brush_free(MyPaintBrush *self)
{
    for (int i=0; i<MYPAINT_BRUSH_SETTINGS_COUNT; i++) {
        mapping_free(self->settings[i]);
    }
    rng_double_free (self->rng);
    self->rng = NULL;

#ifdef HAVE_JSON_C
    json_object_put(self->brush_json);
#endif

    free(self);
}

/**
  * mypaint_brush_unref: (skip)
  *
  * Decrease the reference count. Will be freed when it hits 0.
  */
void
mypaint_brush_unref(MyPaintBrush *self)
{
    self->refcount--;
    if (self->refcount == 0) {
        brush_free(self);
    }
}
/**
  * mypaint_brush_ref: (skip)
  *
  * Increase the reference count.
  */
void
mypaint_brush_ref(MyPaintBrush *self)
{
    self->refcount++;
}

/**
  * mypaint_brush_get_total_stroke_painting_time:
  *
  * Return the total amount of painting time for the current stroke.
  */
double
mypaint_brush_get_total_stroke_painting_time(MyPaintBrush *self)
{
    return self->stroke_total_painting_time;
}

/**
  * mypaint_brush_set_print_inputs:
  *
  * Enable/Disable printing of brush engine inputs on stderr. Intended for debugging only.
  */
void
mypaint_brush_set_print_inputs(MyPaintBrush *self, gboolean enabled)
{
    self->print_inputs = enabled;
}

/**
  * mypaint_brush_reset:
  *
  * Reset the current brush engine state.
  * Used when the next mypaint_brush_stroke_to() call is not related to the current state.
  * Note that the reset request is queued and changes in state will only happen on next stroke_to()
  */
void
mypaint_brush_reset(MyPaintBrush *self)
{
    self->reset_requested = TRUE;
}

/**
  * mypaint_brush_new_stroke:
  *
  * Start a new stroke.
  */
void
mypaint_brush_new_stroke(MyPaintBrush *self)
{
    self->stroke_current_idling_time = 0;
    self->stroke_total_painting_time = 0;
}

/**
  * mypaint_brush_set_base_value:
  *
  * Set the base value of a brush setting.
  */
void
mypaint_brush_set_base_value(MyPaintBrush *self, MyPaintBrushSetting id, float value)
{
    assert (id >= 0 && id < MYPAINT_BRUSH_SETTINGS_COUNT);
    mapping_set_base_value(self->settings[id], value);

    settings_base_values_have_changed (self);
}

/**
  * mypaint_brush_get_base_value:
  *
  * Get the base value of a brush setting.
  */
float
mypaint_brush_get_base_value(MyPaintBrush *self, MyPaintBrushSetting id)
{
    assert (id >= 0 && id < MYPAINT_BRUSH_SETTINGS_COUNT);
    return mapping_get_base_value(self->settings[id]);
}

/**
  * mypaint_brush_set_mapping_n:
  *
  * Set the number of points used for the dynamics mapping between a #MyPaintBrushInput and #MyPaintBrushSetting.
  */
void
mypaint_brush_set_mapping_n(MyPaintBrush *self, MyPaintBrushSetting id, MyPaintBrushInput input, int n)
{
    assert (id >= 0 && id < MYPAINT_BRUSH_SETTINGS_COUNT);
    mapping_set_n(self->settings[id], input, n);
}

/**
  * mypaint_brush_get_mapping_n:
  *
  * Get the number of points used for the dynamics mapping between a #MyPaintBrushInput and #MyPaintBrushSetting.
  */
int
mypaint_brush_get_mapping_n(MyPaintBrush *self, MyPaintBrushSetting id, MyPaintBrushInput input)
{
    return mapping_get_n(self->settings[id], input);
}

/**
  * mypaint_brush_is_constant:
  *
  * Returns TRUE if the brush has no dynamics for the given #MyPaintBrushSetting
  */
gboolean
mypaint_brush_is_constant(MyPaintBrush *self, MyPaintBrushSetting id)
{
    assert (id >= 0 && id < MYPAINT_BRUSH_SETTINGS_COUNT);
    return mapping_is_constant(self->settings[id]);
}

/**
  * mypaint_brush_get_inputs_used_n:
  *
  * Returns how many inputs are used for the dynamics of a #MyPaintBrushSetting
  */
int
mypaint_brush_get_inputs_used_n(MyPaintBrush *self, MyPaintBrushSetting id)
{
    assert (id >= 0 && id < MYPAINT_BRUSH_SETTINGS_COUNT);
    return mapping_get_inputs_used_n(self->settings[id]);
}

/**
  * mypaint_brush_set_mapping_point:
  *
  * Set a X,Y point of a dynamics mapping.
  * The index must be within the number of points set using mypaint_brush_set_mapping_n()
  */
void
mypaint_brush_set_mapping_point(MyPaintBrush *self, MyPaintBrushSetting id, MyPaintBrushInput input, int index, float x, float y)
{
    assert (id >= 0 && id < MYPAINT_BRUSH_SETTINGS_COUNT);
    mapping_set_point(self->settings[id], input, index, x, y);
}

/**
 * mypaint_brush_get_mapping_point:
 * @x: (out): Location to return the X value
 * @y: (out): Location to return the Y value
 *
 * Get a X,Y point of a dynamics mapping.
 **/
void
mypaint_brush_get_mapping_point(MyPaintBrush *self, MyPaintBrushSetting id, MyPaintBrushInput input, int index, float *x, float *y)
{
    assert (id >= 0 && id < MYPAINT_BRUSH_SETTINGS_COUNT);
    mapping_get_point(self->settings[id], input, index, x, y);
}

/**
 * mypaint_brush_get_state:
 *
 * Get an internal brush engine state.
 * Normally used for debugging, but can be used to implement record & replay functionality.
 **/
float
mypaint_brush_get_state(MyPaintBrush *self, MyPaintBrushState i)
{
    assert (i >= 0 && i < MYPAINT_BRUSH_STATES_COUNT);
    return self->states[i];
}

/**
 * mypaint_brush_set_state:
 *
 * Set an internal brush engine state.
 * Normally used for debugging, but can be used to implement record & replay functionality.
 **/
void
mypaint_brush_set_state(MyPaintBrush *self, MyPaintBrushState i, float value)
{
    assert (i >= 0 && i < MYPAINT_BRUSH_STATES_COUNT);
    self->states[i] = value;
}


// Returns the smallest angular difference (counterclockwise or clockwise) a to b, in degrees.
// Clockwise is positive.
static inline float
smallest_angular_difference(float a, float b)
{
    float d_cw, d_ccw;
    a = fmodf(a, 360.0);
    b = fmodf(b, 360.0);
    if (a > b) {
        d_cw = a - b;
        d_ccw = b + 360.0 - a;
    }
    else {
        d_cw = a + 360.0 - b;
        d_ccw = b - a;
    }
    return (d_cw < d_ccw) ? -d_cw : d_ccw;
}


  // returns the fraction still left after t seconds
  float exp_decay (float T_const, float t)
  {
    // the argument might not make mathematical sense (whatever.)
    if (T_const <= 0.001) {
      return 0.0;
    }

    float arg = -t / T_const;
    return expf(arg);
  }


  void settings_base_values_have_changed (MyPaintBrush *self)
  {
    // precalculate stuff that does not change dynamically

    // Precalculate how the physical speed will be mapped to the speed input value.
    // The forumla for this mapping is:
    //
    // y = log(gamma+x)*m + q;
    //
    // x: the physical speed (pixels per basic dab radius)
    // y: the speed input that will be reported
    // gamma: parameter set by ths user (small means a logarithmic mapping, big linear)
    // m, q: parameters to scale and translate the curve
    //
    // The code below calculates m and q given gamma and two hardcoded constraints.
    //
    int i=0;
    for (i=0; i<2; i++) {
      float gamma;
      gamma = mapping_get_base_value(self->settings[(i==0)?MYPAINT_BRUSH_SETTING_SPEED1_GAMMA:MYPAINT_BRUSH_SETTING_SPEED2_GAMMA]);
      gamma = expf(gamma);

      float fix1_x, fix1_y, fix2_x, fix2_dy;
      fix1_x = 45.0;
      fix1_y = 0.5;
      fix2_x = 45.0;
      fix2_dy = 0.015;

      float m, q;
      float c1;
      c1 = log(fix1_x+gamma);
      m = fix2_dy * (fix2_x + gamma);
      q = fix1_y - m*c1;

      self->speed_mapping_gamma[i] = gamma;
      self->speed_mapping_m[i] = m;
      self->speed_mapping_q[i] = q;
    }
  }

  // This function runs a brush "simulation" step. Usually it is
  // called once or twice per dab. In theory the precision of the
  // "simulation" gets better when it is called more often. In
  // practice this only matters if there are some highly nonlinear
  // mappings in critical places or extremely few events per second.
  //
  // note: parameters are is dx/ddab, ..., dtime/ddab (dab is the number, 5.0 = 5th dab)
  void update_states_and_setting_values (MyPaintBrush *self, float step_dx, float step_dy, float step_dpressure, float step_declination, float step_ascension, float step_dtime)
  {
    float pressure;
    float inputs[MYPAINT_BRUSH_INPUTS_COUNT];

    if (step_dtime < 0.0) {
      printf("Time is running backwards!\n");
      step_dtime = 0.001;
    } else if (step_dtime == 0.0) {
      // FIXME: happens about every 10th start, workaround (against division by zero)
      step_dtime = 0.001;
    }

    self->states[MYPAINT_BRUSH_STATE_X]        += step_dx;
    self->states[MYPAINT_BRUSH_STATE_Y]        += step_dy;
    self->states[MYPAINT_BRUSH_STATE_PRESSURE] += step_dpressure;

    self->states[MYPAINT_BRUSH_STATE_DECLINATION] += step_declination;
    self->states[MYPAINT_BRUSH_STATE_ASCENSION] += step_ascension;

    float base_radius = expf(mapping_get_base_value(self->settings[MYPAINT_BRUSH_SETTING_RADIUS_LOGARITHMIC]));

    // FIXME: does happen (interpolation problem?)
    if (self->states[MYPAINT_BRUSH_STATE_PRESSURE] <= 0.0) self->states[MYPAINT_BRUSH_STATE_PRESSURE] = 0.0;
    pressure = self->states[MYPAINT_BRUSH_STATE_PRESSURE];

    { // start / end stroke (for "stroke" input only)
      if (!self->states[MYPAINT_BRUSH_STATE_STROKE_STARTED]) {
        if (pressure > mapping_get_base_value(self->settings[MYPAINT_BRUSH_SETTING_STROKE_THRESHOLD]) + 0.0001) {
          // start new stroke
          //printf("stroke start %f\n", pressure);
          self->states[MYPAINT_BRUSH_STATE_STROKE_STARTED] = 1;
          self->states[MYPAINT_BRUSH_STATE_STROKE] = 0.0;
        }
      } else {
        if (pressure <= mapping_get_base_value(self->settings[MYPAINT_BRUSH_SETTING_STROKE_THRESHOLD]) * 0.9 + 0.0001) {
          // end stroke
          //printf("stroke end\n");
          self->states[MYPAINT_BRUSH_STATE_STROKE_STARTED] = 0;
        }
      }
    }

    // now follows input handling

    float norm_dx, norm_dy, norm_dist, norm_speed;
    norm_dx = step_dx / step_dtime / base_radius;
    norm_dy = step_dy / step_dtime / base_radius;
    norm_speed = sqrt(SQR(norm_dx) + SQR(norm_dy));
    norm_dist = norm_speed * step_dtime;

    inputs[MYPAINT_BRUSH_INPUT_PRESSURE] = pressure * expf(mapping_get_base_value(self->settings[MYPAINT_BRUSH_SETTING_PRESSURE_GAIN_LOG]));
    inputs[MYPAINT_BRUSH_INPUT_SPEED1] = log(self->speed_mapping_gamma[0] + self->states[MYPAINT_BRUSH_STATE_NORM_SPEED1_SLOW])*self->speed_mapping_m[0] + self->speed_mapping_q[0];
    inputs[MYPAINT_BRUSH_INPUT_SPEED2] = log(self->speed_mapping_gamma[1] + self->states[MYPAINT_BRUSH_STATE_NORM_SPEED2_SLOW])*self->speed_mapping_m[1] + self->speed_mapping_q[1];
    inputs[MYPAINT_BRUSH_INPUT_RANDOM] = rng_double_next(self->rng);
    inputs[MYPAINT_BRUSH_INPUT_STROKE] = MIN(self->states[MYPAINT_BRUSH_STATE_STROKE], 1.0);
    inputs[MYPAINT_BRUSH_INPUT_DIRECTION] = fmodf (atan2f (self->states[MYPAINT_BRUSH_STATE_DIRECTION_DY], self->states[MYPAINT_BRUSH_STATE_DIRECTION_DX])/(2*M_PI)*360 + 180.0, 180.0);
    inputs[MYPAINT_BRUSH_INPUT_TILT_DECLINATION] = self->states[MYPAINT_BRUSH_STATE_DECLINATION];
    inputs[MYPAINT_BRUSH_INPUT_TILT_ASCENSION] = fmodf(self->states[MYPAINT_BRUSH_STATE_ASCENSION] + 180.0, 360.0) - 180.0;

    inputs[MYPAINT_BRUSH_INPUT_CUSTOM] = self->states[MYPAINT_BRUSH_STATE_CUSTOM_INPUT];
    if (self->print_inputs) {
      printf("press=% 4.3f, speed1=% 4.4f\tspeed2=% 4.4f\tstroke=% 4.3f\tcustom=% 4.3f\n", (double)inputs[MYPAINT_BRUSH_INPUT_PRESSURE], (double)inputs[MYPAINT_BRUSH_INPUT_SPEED1], (double)inputs[MYPAINT_BRUSH_INPUT_SPEED2], (double)inputs[MYPAINT_BRUSH_INPUT_STROKE], (double)inputs[MYPAINT_BRUSH_INPUT_CUSTOM]);
    }
    // FIXME: this one fails!!!
    //assert(inputs[MYPAINT_BRUSH_INPUT_SPEED1] >= 0.0 && inputs[MYPAINT_BRUSH_INPUT_SPEED1] < 1e8); // checking for inf

    int i=0;
    for (i=0; i<MYPAINT_BRUSH_SETTINGS_COUNT; i++) {
      self->settings_value[i] = mapping_calculate(self->settings[i], (inputs));
    }

    {
      float fac = 1.0 - exp_decay (self->settings_value[MYPAINT_BRUSH_SETTING_SLOW_TRACKING_PER_DAB], 1.0);
      self->states[MYPAINT_BRUSH_STATE_ACTUAL_X] += (self->states[MYPAINT_BRUSH_STATE_X] - self->states[MYPAINT_BRUSH_STATE_ACTUAL_X]) * fac; // FIXME: should this depend on base radius?
      self->states[MYPAINT_BRUSH_STATE_ACTUAL_Y] += (self->states[MYPAINT_BRUSH_STATE_Y] - self->states[MYPAINT_BRUSH_STATE_ACTUAL_Y]) * fac;
    }

    { // slow speed
      float fac;
      fac = 1.0 - exp_decay (self->settings_value[MYPAINT_BRUSH_SETTING_SPEED1_SLOWNESS], step_dtime);
      self->states[MYPAINT_BRUSH_STATE_NORM_SPEED1_SLOW] += (norm_speed - self->states[MYPAINT_BRUSH_STATE_NORM_SPEED1_SLOW]) * fac;
      fac = 1.0 - exp_decay (self->settings_value[MYPAINT_BRUSH_SETTING_SPEED2_SLOWNESS], step_dtime);
      self->states[MYPAINT_BRUSH_STATE_NORM_SPEED2_SLOW] += (norm_speed - self->states[MYPAINT_BRUSH_STATE_NORM_SPEED2_SLOW]) * fac;
    }

    { // slow speed, but as vector this time

      // FIXME: offset_by_speed should be removed.
      //   Is it broken, non-smooth, system-dependent math?!
      //   A replacement could be a directed random offset.

      float time_constant = expf(self->settings_value[MYPAINT_BRUSH_SETTING_OFFSET_BY_SPEED_SLOWNESS]*0.01)-1.0;
      // Workaround for a bug that happens mainly on Windows, causing
      // individual dabs to be placed far far away. Using the speed
      // with zero filtering is just asking for trouble anyway.
      if (time_constant < 0.002) time_constant = 0.002;
      float fac = 1.0 - exp_decay (time_constant, step_dtime);
      self->states[MYPAINT_BRUSH_STATE_NORM_DX_SLOW] += (norm_dx - self->states[MYPAINT_BRUSH_STATE_NORM_DX_SLOW]) * fac;
      self->states[MYPAINT_BRUSH_STATE_NORM_DY_SLOW] += (norm_dy - self->states[MYPAINT_BRUSH_STATE_NORM_DY_SLOW]) * fac;
    }

    { // orientation (similar lowpass filter as above, but use dabtime instead of wallclock time)
      float dx = step_dx / base_radius;
      float dy = step_dy / base_radius;
      float step_in_dabtime = hypotf(dx, dy); // FIXME: are we recalculating something here that we already have?
      float fac = 1.0 - exp_decay (exp(self->settings_value[MYPAINT_BRUSH_SETTING_DIRECTION_FILTER]*0.5)-1.0, step_in_dabtime);

      float dx_old = self->states[MYPAINT_BRUSH_STATE_DIRECTION_DX];
      float dy_old = self->states[MYPAINT_BRUSH_STATE_DIRECTION_DY];
      // use the opposite speed vector if it is closer (we don't care about 180 degree turns)
      if (SQR(dx_old-dx) + SQR(dy_old-dy) > SQR(dx_old-(-dx)) + SQR(dy_old-(-dy))) {
        dx = -dx;
        dy = -dy;
      }
      self->states[MYPAINT_BRUSH_STATE_DIRECTION_DX] += (dx - self->states[MYPAINT_BRUSH_STATE_DIRECTION_DX]) * fac;
      self->states[MYPAINT_BRUSH_STATE_DIRECTION_DY] += (dy - self->states[MYPAINT_BRUSH_STATE_DIRECTION_DY]) * fac;
    }

    { // custom input
      float fac;
      fac = 1.0 - exp_decay (self->settings_value[MYPAINT_BRUSH_SETTING_CUSTOM_INPUT_SLOWNESS], 0.1);
      self->states[MYPAINT_BRUSH_STATE_CUSTOM_INPUT] += (self->settings_value[MYPAINT_BRUSH_SETTING_CUSTOM_INPUT] - self->states[MYPAINT_BRUSH_STATE_CUSTOM_INPUT]) * fac;
    }

    { // stroke length
      float frequency;
      float wrap;
      frequency = expf(-self->settings_value[MYPAINT_BRUSH_SETTING_STROKE_DURATION_LOGARITHMIC]);
      self->states[MYPAINT_BRUSH_STATE_STROKE] += norm_dist * frequency;
      // can happen, probably caused by rounding
      if (self->states[MYPAINT_BRUSH_STATE_STROKE] < 0) self->states[MYPAINT_BRUSH_STATE_STROKE] = 0;
      wrap = 1.0 + self->settings_value[MYPAINT_BRUSH_SETTING_STROKE_HOLDTIME];
      if (self->states[MYPAINT_BRUSH_STATE_STROKE] > wrap) {
        if (wrap > 9.9 + 1.0) {
          // "inifinity", just hold stroke somewhere >= 1.0
          self->states[MYPAINT_BRUSH_STATE_STROKE] = 1.0;
        } else {
          self->states[MYPAINT_BRUSH_STATE_STROKE] = fmodf(self->states[MYPAINT_BRUSH_STATE_STROKE], wrap);
          // just in case
          if (self->states[MYPAINT_BRUSH_STATE_STROKE] < 0) self->states[MYPAINT_BRUSH_STATE_STROKE] = 0;
        }
      }
    }

    // calculate final radius
    float radius_log;
    radius_log = self->settings_value[MYPAINT_BRUSH_SETTING_RADIUS_LOGARITHMIC];
    self->states[MYPAINT_BRUSH_STATE_ACTUAL_RADIUS] = expf(radius_log);
    if (self->states[MYPAINT_BRUSH_STATE_ACTUAL_RADIUS] < ACTUAL_RADIUS_MIN) self->states[MYPAINT_BRUSH_STATE_ACTUAL_RADIUS] = ACTUAL_RADIUS_MIN;
    if (self->states[MYPAINT_BRUSH_STATE_ACTUAL_RADIUS] > ACTUAL_RADIUS_MAX) self->states[MYPAINT_BRUSH_STATE_ACTUAL_RADIUS] = ACTUAL_RADIUS_MAX;

    // aspect ratio (needs to be caluclated here because it can affect the dab spacing)
    self->states[MYPAINT_BRUSH_STATE_ACTUAL_ELLIPTICAL_DAB_RATIO] = self->settings_value[MYPAINT_BRUSH_SETTING_ELLIPTICAL_DAB_RATIO];
    self->states[MYPAINT_BRUSH_STATE_ACTUAL_ELLIPTICAL_DAB_ANGLE] = self->settings_value[MYPAINT_BRUSH_SETTING_ELLIPTICAL_DAB_ANGLE];
  }

  // Called only from stroke_to(). Calculate everything needed to
  // draw the dab, then let the surface do the actual drawing.
  //
  // This is only gets called right after update_states_and_setting_values().
  // Returns TRUE if the surface was modified.
  gboolean prepare_and_draw_dab (MyPaintBrush *self, MyPaintSurface * surface)
  {
    float x, y, opaque;
    float radius;

    // ensure we don't get a positive result with two negative opaque values
    if (self->settings_value[MYPAINT_BRUSH_SETTING_OPAQUE] < 0) self->settings_value[MYPAINT_BRUSH_SETTING_OPAQUE] = 0;
    opaque = self->settings_value[MYPAINT_BRUSH_SETTING_OPAQUE] * self->settings_value[MYPAINT_BRUSH_SETTING_OPAQUE_MULTIPLY];
    opaque = CLAMP(opaque, 0.0, 1.0);
    //if (opaque == 0.0) return FALSE; <-- cannot do that, since we need to update smudge state.
    if (self->settings_value[MYPAINT_BRUSH_SETTING_OPAQUE_LINEARIZE]) {
      // OPTIMIZE: no need to recalculate this for each dab
      float alpha, beta, alpha_dab, beta_dab;
      float dabs_per_pixel;
      // dabs_per_pixel is just estimated roughly, I didn't think hard
      // about the case when the radius changes during the stroke
      dabs_per_pixel = (
                        mapping_get_base_value(self->settings[MYPAINT_BRUSH_SETTING_DABS_PER_ACTUAL_RADIUS]) +
                        mapping_get_base_value(self->settings[MYPAINT_BRUSH_SETTING_DABS_PER_BASIC_RADIUS])
                        ) * 2.0;

      // the correction is probably not wanted if the dabs don't overlap
      if (dabs_per_pixel < 1.0) dabs_per_pixel = 1.0;

      // interpret the user-setting smoothly
      dabs_per_pixel = 1.0 + mapping_get_base_value(self->settings[MYPAINT_BRUSH_SETTING_OPAQUE_LINEARIZE])*(dabs_per_pixel-1.0);

      // see doc/brushdab_saturation.png
      //      beta = beta_dab^dabs_per_pixel
      // <==> beta_dab = beta^(1/dabs_per_pixel)
      alpha = opaque;
      beta = 1.0-alpha;
      beta_dab = powf(beta, 1.0/dabs_per_pixel);
      alpha_dab = 1.0-beta_dab;
      opaque = alpha_dab;
    }

    x = self->states[MYPAINT_BRUSH_STATE_ACTUAL_X];
    y = self->states[MYPAINT_BRUSH_STATE_ACTUAL_Y];

    float base_radius = expf(mapping_get_base_value(self->settings[MYPAINT_BRUSH_SETTING_RADIUS_LOGARITHMIC]));

    if (self->settings_value[MYPAINT_BRUSH_SETTING_OFFSET_BY_SPEED]) {
      x += self->states[MYPAINT_BRUSH_STATE_NORM_DX_SLOW] * self->settings_value[MYPAINT_BRUSH_SETTING_OFFSET_BY_SPEED] * 0.1 * base_radius;
      y += self->states[MYPAINT_BRUSH_STATE_NORM_DY_SLOW] * self->settings_value[MYPAINT_BRUSH_SETTING_OFFSET_BY_SPEED] * 0.1 * base_radius;
    }

    if (self->settings_value[MYPAINT_BRUSH_SETTING_OFFSET_BY_RANDOM]) {
      float amp = self->settings_value[MYPAINT_BRUSH_SETTING_OFFSET_BY_RANDOM];
      if (amp < 0.0) amp = 0.0;
      x += rand_gauss (self->rng) * amp * base_radius;
      y += rand_gauss (self->rng) * amp * base_radius;
    }


    radius = self->states[MYPAINT_BRUSH_STATE_ACTUAL_RADIUS];
    if (self->settings_value[MYPAINT_BRUSH_SETTING_RADIUS_BY_RANDOM]) {
      float radius_log, alpha_correction;
      // go back to logarithmic radius to add the noise
      radius_log  = self->settings_value[MYPAINT_BRUSH_SETTING_RADIUS_LOGARITHMIC];
      radius_log += rand_gauss (self->rng) * self->settings_value[MYPAINT_BRUSH_SETTING_RADIUS_BY_RANDOM];
      radius = expf(radius_log);
      radius = CLAMP(radius, ACTUAL_RADIUS_MIN, ACTUAL_RADIUS_MAX);
      alpha_correction = self->states[MYPAINT_BRUSH_STATE_ACTUAL_RADIUS] / radius;
      alpha_correction = SQR(alpha_correction);
      if (alpha_correction <= 1.0) {
        opaque *= alpha_correction;
      }
    }

    // update smudge color
    if (self->settings_value[MYPAINT_BRUSH_SETTING_SMUDGE_LENGTH] < 1.0 &&
        // optimization, since normal brushes have smudge_length == 0.5 without actually smudging
        (self->settings_value[MYPAINT_BRUSH_SETTING_SMUDGE] != 0.0 || !mapping_is_constant(self->settings[MYPAINT_BRUSH_SETTING_SMUDGE]))) {

      float fac = self->settings_value[MYPAINT_BRUSH_SETTING_SMUDGE_LENGTH];
      if (fac < 0.01) fac = 0.01;
      int px, py;
      px = ROUND(x);
      py = ROUND(y);

      // Calling get_color() is almost as expensive as rendering a
      // dab. Because of this we use the previous value if it is not
      // expected to hurt quality too much. We call it at most every
      // second dab.
      float r, g, b, a;
      self->states[MYPAINT_BRUSH_STATE_LAST_GETCOLOR_RECENTNESS] *= fac;
      if (self->states[MYPAINT_BRUSH_STATE_LAST_GETCOLOR_RECENTNESS] < 0.5*fac) {
        if (self->states[MYPAINT_BRUSH_STATE_LAST_GETCOLOR_RECENTNESS] == 0.0) {
          // first initialization of smudge color
          fac = 0.0;
        }
        self->states[MYPAINT_BRUSH_STATE_LAST_GETCOLOR_RECENTNESS] = 1.0;

        float smudge_radius = radius * expf(self->settings_value[MYPAINT_BRUSH_SETTING_SMUDGE_RADIUS_LOG]);
        smudge_radius = CLAMP(smudge_radius, ACTUAL_RADIUS_MIN, ACTUAL_RADIUS_MAX);
        mypaint_surface_get_color(surface, px, py, smudge_radius, &r, &g, &b, &a);

        self->states[MYPAINT_BRUSH_STATE_LAST_GETCOLOR_R] = r;
        self->states[MYPAINT_BRUSH_STATE_LAST_GETCOLOR_G] = g;
        self->states[MYPAINT_BRUSH_STATE_LAST_GETCOLOR_B] = b;
        self->states[MYPAINT_BRUSH_STATE_LAST_GETCOLOR_A] = a;
      } else {
        r = self->states[MYPAINT_BRUSH_STATE_LAST_GETCOLOR_R];
        g = self->states[MYPAINT_BRUSH_STATE_LAST_GETCOLOR_G];
        b = self->states[MYPAINT_BRUSH_STATE_LAST_GETCOLOR_B];
        a = self->states[MYPAINT_BRUSH_STATE_LAST_GETCOLOR_A];
      }

      // updated the smudge color (stored with premultiplied alpha)
      self->states[MYPAINT_BRUSH_STATE_SMUDGE_A ] = fac*self->states[MYPAINT_BRUSH_STATE_SMUDGE_A ] + (1-fac)*a;
      // fix rounding errors
      self->states[MYPAINT_BRUSH_STATE_SMUDGE_A ] = CLAMP(self->states[MYPAINT_BRUSH_STATE_SMUDGE_A], 0.0, 1.0);

      self->states[MYPAINT_BRUSH_STATE_SMUDGE_RA] = fac*self->states[MYPAINT_BRUSH_STATE_SMUDGE_RA] + (1-fac)*r*a;
      self->states[MYPAINT_BRUSH_STATE_SMUDGE_GA] = fac*self->states[MYPAINT_BRUSH_STATE_SMUDGE_GA] + (1-fac)*g*a;
      self->states[MYPAINT_BRUSH_STATE_SMUDGE_BA] = fac*self->states[MYPAINT_BRUSH_STATE_SMUDGE_BA] + (1-fac)*b*a;
    }

    // color part

    float color_h = mapping_get_base_value(self->settings[MYPAINT_BRUSH_SETTING_COLOR_H]);
    float color_s = mapping_get_base_value(self->settings[MYPAINT_BRUSH_SETTING_COLOR_S]);
    float color_v = mapping_get_base_value(self->settings[MYPAINT_BRUSH_SETTING_COLOR_V]);
    float eraser_target_alpha = 1.0;
    if (self->settings_value[MYPAINT_BRUSH_SETTING_SMUDGE] > 0.0) {
      // mix (in RGB) the smudge color with the brush color
      hsv_to_rgb_float (&color_h, &color_s, &color_v);
      float fac = self->settings_value[MYPAINT_BRUSH_SETTING_SMUDGE];
      if (fac > 1.0) fac = 1.0;
      // If the smudge color somewhat transparent, then the resulting
      // dab will do erasing towards that transparency level.
      // see also ../doc/smudge_math.png
      eraser_target_alpha = (1-fac)*1.0 + fac*self->states[MYPAINT_BRUSH_STATE_SMUDGE_A];
      // fix rounding errors (they really seem to happen in the previous line)
      eraser_target_alpha = CLAMP(eraser_target_alpha, 0.0, 1.0);
      if (eraser_target_alpha > 0) {
        color_h = (fac*self->states[MYPAINT_BRUSH_STATE_SMUDGE_RA] + (1-fac)*color_h) / eraser_target_alpha;
        color_s = (fac*self->states[MYPAINT_BRUSH_STATE_SMUDGE_GA] + (1-fac)*color_s) / eraser_target_alpha;
        color_v = (fac*self->states[MYPAINT_BRUSH_STATE_SMUDGE_BA] + (1-fac)*color_v) / eraser_target_alpha;
      } else {
        // we are only erasing; the color does not matter
        color_h = 1.0;
        color_s = 0.0;
        color_v = 0.0;
      }
      rgb_to_hsv_float (&color_h, &color_s, &color_v);
    }

    // eraser
    if (self->settings_value[MYPAINT_BRUSH_SETTING_ERASER]) {
      eraser_target_alpha *= (1.0-self->settings_value[MYPAINT_BRUSH_SETTING_ERASER]);
    }

    // HSV color change
    color_h += self->settings_value[MYPAINT_BRUSH_SETTING_CHANGE_COLOR_H];
    color_s += self->settings_value[MYPAINT_BRUSH_SETTING_CHANGE_COLOR_HSV_S];
    color_v += self->settings_value[MYPAINT_BRUSH_SETTING_CHANGE_COLOR_V];

    // HSL color change
    if (self->settings_value[MYPAINT_BRUSH_SETTING_CHANGE_COLOR_L] || self->settings_value[MYPAINT_BRUSH_SETTING_CHANGE_COLOR_HSL_S]) {
      // (calculating way too much here, can be optimized if neccessary)
      // this function will CLAMP the inputs
      hsv_to_rgb_float (&color_h, &color_s, &color_v);
      rgb_to_hsl_float (&color_h, &color_s, &color_v);
      color_v += self->settings_value[MYPAINT_BRUSH_SETTING_CHANGE_COLOR_L];
      color_s += self->settings_value[MYPAINT_BRUSH_SETTING_CHANGE_COLOR_HSL_S];
      hsl_to_rgb_float (&color_h, &color_s, &color_v);
      rgb_to_hsv_float (&color_h, &color_s, &color_v);
    }

    float hardness = CLAMP(self->settings_value[MYPAINT_BRUSH_SETTING_HARDNESS], 0.0, 1.0);

    // anti-aliasing attempt (works surprisingly well for ink brushes)
    float current_fadeout_in_pixels = radius * (1.0 - hardness);
    float min_fadeout_in_pixels = self->settings_value[MYPAINT_BRUSH_SETTING_ANTI_ALIASING];
    if (current_fadeout_in_pixels < min_fadeout_in_pixels) {
      // need to soften the brush (decrease hardness), but keep optical radius
      // so we tune both radius and hardness, to get the desired fadeout_in_pixels
      float current_optical_radius = radius - (1.0-hardness)*radius/2.0;

      // Equation 1: (new fadeout must be equal to min_fadeout)
      //   min_fadeout_in_pixels = radius_new*(1.0 - hardness_new)
      // Equation 2: (optical radius must remain unchanged)
      //   current_optical_radius = radius_new - (1.0-hardness_new)*radius_new/2.0
      //
      // Solved Equation 1 for hardness_new, using Equation 2: (thanks to mathomatic)
      float hardness_new = ((current_optical_radius - (min_fadeout_in_pixels/2.0))/(current_optical_radius + (min_fadeout_in_pixels/2.0)));
      // Using Equation 1:
      float radius_new = (min_fadeout_in_pixels/(1.0 - hardness_new));

      hardness = hardness_new;
      radius = radius_new;
    }

    // snap to pixel
    float snapToPixel = self->settings_value[MYPAINT_BRUSH_SETTING_SNAP_TO_PIXEL];
    if (snapToPixel > 0.0)
    {
      // linear interpolation between non-snapped and snapped
      float snapped_x = floor(x) + 0.5;
      float snapped_y = floor(y) + 0.5;
      x = x + (snapped_x - x) * snapToPixel;
      y = y + (snapped_y - y) * snapToPixel;

      float snapped_radius = roundf(radius * 2.0) / 2.0;
      if (snapped_radius < 0.5)
        snapped_radius = 0.5;

      if (snapToPixel > 0.9999 )
      {
        snapped_radius -= 0.0001; // this fixes precision issues where
                                  // neighboor pixels could be wrongly painted
      }

      radius = radius + (snapped_radius - radius) * snapToPixel;
    }

    // the functions below will CLAMP most inputs
    hsv_to_rgb_float (&color_h, &color_s, &color_v);
    return mypaint_surface_draw_dab (surface, x, y, radius, color_h, color_s, color_v, opaque, hardness, eraser_target_alpha,
                              self->states[MYPAINT_BRUSH_STATE_ACTUAL_ELLIPTICAL_DAB_RATIO], self->states[MYPAINT_BRUSH_STATE_ACTUAL_ELLIPTICAL_DAB_ANGLE],
                              self->settings_value[MYPAINT_BRUSH_SETTING_LOCK_ALPHA],
                              self->settings_value[MYPAINT_BRUSH_SETTING_COLORIZE]);
  }

  // How many dabs will be drawn between the current and the next (x, y, pressure, +dt) position?
  float count_dabs_to (MyPaintBrush *self, float x, float y, float pressure, float dt)
  {
    float xx, yy;
    float res1, res2, res3;
    float dist;

    if (self->states[MYPAINT_BRUSH_STATE_ACTUAL_RADIUS] == 0.0) self->states[MYPAINT_BRUSH_STATE_ACTUAL_RADIUS] = expf(mapping_get_base_value(self->settings[MYPAINT_BRUSH_SETTING_RADIUS_LOGARITHMIC]));
    if (self->states[MYPAINT_BRUSH_STATE_ACTUAL_RADIUS] < ACTUAL_RADIUS_MIN) self->states[MYPAINT_BRUSH_STATE_ACTUAL_RADIUS] = ACTUAL_RADIUS_MIN;
    if (self->states[MYPAINT_BRUSH_STATE_ACTUAL_RADIUS] > ACTUAL_RADIUS_MAX) self->states[MYPAINT_BRUSH_STATE_ACTUAL_RADIUS] = ACTUAL_RADIUS_MAX;


    // OPTIMIZE: expf() called too often
    float base_radius = expf(mapping_get_base_value(self->settings[MYPAINT_BRUSH_SETTING_RADIUS_LOGARITHMIC]));
    if (base_radius < ACTUAL_RADIUS_MIN) base_radius = ACTUAL_RADIUS_MIN;
    if (base_radius > ACTUAL_RADIUS_MAX) base_radius = ACTUAL_RADIUS_MAX;
    //if (base_radius < 0.5) base_radius = 0.5;
    //if (base_radius > 500.0) base_radius = 500.0;

    xx = x - self->states[MYPAINT_BRUSH_STATE_X];
    yy = y - self->states[MYPAINT_BRUSH_STATE_Y];
    //dp = pressure - pressure; // Not useful?
    // TODO: control rate with pressure (dabs per pressure) (dpressure is useless)

    if (self->states[MYPAINT_BRUSH_STATE_ACTUAL_ELLIPTICAL_DAB_RATIO] > 1.0) {
      // code duplication, see tiledsurface::draw_dab()
      float angle_rad=self->states[MYPAINT_BRUSH_STATE_ACTUAL_ELLIPTICAL_DAB_ANGLE]/360*2*M_PI;
      float cs=cos(angle_rad);
      float sn=sin(angle_rad);
      float yyr=(yy*cs-xx*sn)*self->states[MYPAINT_BRUSH_STATE_ACTUAL_ELLIPTICAL_DAB_RATIO];
      float xxr=yy*sn+xx*cs;
      dist = sqrt(yyr*yyr + xxr*xxr);
    } else {
      dist = hypotf(xx, yy);
    }

    // FIXME: no need for base_value or for the range checks above IF always the interpolation
    //        function will be called before this one
    res1 = dist / self->states[MYPAINT_BRUSH_STATE_ACTUAL_RADIUS] * mapping_get_base_value(self->settings[MYPAINT_BRUSH_SETTING_DABS_PER_ACTUAL_RADIUS]);
    res2 = dist / base_radius   * mapping_get_base_value(self->settings[MYPAINT_BRUSH_SETTING_DABS_PER_BASIC_RADIUS]);
    res3 = dt * mapping_get_base_value(self->settings[MYPAINT_BRUSH_SETTING_DABS_PER_SECOND]);
    return res1 + res2 + res3;
  }

  // This function:
  // - is called once for each motion event
  // - does motion event interpolation
  // - paints zero, one or several dabs
  // - decides whether the stroke is finished (for undo/redo)
  // returns TRUE if the stroke is finished or empty
  int mypaint_brush_stroke_to (MyPaintBrush *self, MyPaintSurface *surface,
                                float x, float y, float pressure,
                                float xtilt, float ytilt, double dtime)
  {
    //printf("%f %f %f %f\n", (double)dtime, (double)x, (double)y, (double)pressure);

    float tilt_ascension = 0.0;
    float tilt_declination = 90.0;
    if (xtilt != 0 || ytilt != 0) {
      // shield us from insane tilt input
      xtilt = CLAMP(xtilt, -1.0, 1.0);
      ytilt = CLAMP(ytilt, -1.0, 1.0);
      assert(isfinite(xtilt) && isfinite(ytilt));

      tilt_ascension = 180.0*atan2(-xtilt, ytilt)/M_PI;
      float e;
      if (abs(xtilt) > abs(ytilt)) {
        e = sqrt(1+ytilt*ytilt);
      } else {
        e = sqrt(1+xtilt*xtilt);
      }
      float rad = hypot(xtilt, ytilt);
      float cos_alpha = rad/e;
      if (cos_alpha >= 1.0) cos_alpha = 1.0; // fix numerical inaccuracy
      tilt_declination = 180.0*acos(cos_alpha)/M_PI;

      assert(isfinite(tilt_ascension));
      assert(isfinite(tilt_declination));
    }

    // printf("xtilt %f, ytilt %f\n", (double)xtilt, (double)ytilt);
    // printf("ascension %f, declination %f\n", (double)tilt_ascension, (double)tilt_declination);

    if (pressure <= 0.0) pressure = 0.0;
    if (!isfinite(x) || !isfinite(y) ||
        (x > 1e10 || y > 1e10 || x < -1e10 || y < -1e10)) {
      // workaround attempt for https://gna.org/bugs/?14372
      printf("Warning: ignoring brush::stroke_to with insane inputs (x = %f, y = %f)\n", (double)x, (double)y);
      x = 0.0;
      y = 0.0;
      pressure = 0.0;
    }
    // the assertion below is better than out-of-memory later at save time
    assert(x < 1e8 && y < 1e8 && x > -1e8 && y > -1e8);

    if (dtime < 0) printf("Time jumped backwards by dtime=%f seconds!\n", dtime);
    if (dtime <= 0) dtime = 0.0001; // protect against possible division by zero bugs

    /* way too slow with the new rng, and not working any more anyway...
    rng_double_set_seed (self->rng, self->states[MYPAINT_BRUSH_STATE_RNG_SEED]*0x40000000);
    */

    if (dtime > 0.100 && pressure && self->states[MYPAINT_BRUSH_STATE_PRESSURE] == 0) {
      // Workaround for tablets that don't report motion events without pressure.
      // This is to avoid linear interpolation of the pressure between two events.
      mypaint_brush_stroke_to (self, surface, x, y, 0.0, 90.0, 0.0, dtime-0.0001);
      dtime = 0.0001;
    }

    { // calculate the actual "virtual" cursor position

      // noise first
      if (mapping_get_base_value(self->settings[MYPAINT_BRUSH_SETTING_TRACKING_NOISE])) {
        // OPTIMIZE: expf() called too often
        float base_radius = expf(mapping_get_base_value(self->settings[MYPAINT_BRUSH_SETTING_RADIUS_LOGARITHMIC]));

        x += rand_gauss (self->rng) * mapping_get_base_value(self->settings[MYPAINT_BRUSH_SETTING_TRACKING_NOISE]) * base_radius;
        y += rand_gauss (self->rng) * mapping_get_base_value(self->settings[MYPAINT_BRUSH_SETTING_TRACKING_NOISE]) * base_radius;
      }

      float fac = 1.0 - exp_decay (mapping_get_base_value(self->settings[MYPAINT_BRUSH_SETTING_SLOW_TRACKING]), 100.0*dtime);
      x = self->states[MYPAINT_BRUSH_STATE_X] + (x - self->states[MYPAINT_BRUSH_STATE_X]) * fac;
      y = self->states[MYPAINT_BRUSH_STATE_Y] + (y - self->states[MYPAINT_BRUSH_STATE_Y]) * fac;
    }

    // draw many (or zero) dabs to the next position

    // see doc/stroke2dabs.png
    float dist_moved = self->states[MYPAINT_BRUSH_STATE_DIST];
    float dist_todo = count_dabs_to (self, x, y, pressure, dtime);

    if (dtime > 5 || self->reset_requested) {
      self->reset_requested = FALSE;

      //printf("Brush reset.\n");
      int i=0;
      for (i=0; i<MYPAINT_BRUSH_STATES_COUNT; i++) {
        self->states[i] = 0;
      }

      self->states[MYPAINT_BRUSH_STATE_X] = x;
      self->states[MYPAINT_BRUSH_STATE_Y] = y;
      self->states[MYPAINT_BRUSH_STATE_PRESSURE] = pressure;

      // not resetting, because they will get overwritten below:
      //dx, dy, dpress, dtime

      self->states[MYPAINT_BRUSH_STATE_ACTUAL_X] = self->states[MYPAINT_BRUSH_STATE_X];
      self->states[MYPAINT_BRUSH_STATE_ACTUAL_Y] = self->states[MYPAINT_BRUSH_STATE_Y];
      self->states[MYPAINT_BRUSH_STATE_STROKE] = 1.0; // start in a state as if the stroke was long finished

      return TRUE;
    }

    //g_print("dist = %f\n", states[MYPAINT_BRUSH_STATE_DIST]);
    enum { UNKNOWN, YES, NO } painted = UNKNOWN;
    double dtime_left = dtime;

    float step_dx, step_dy, step_dpressure, step_dtime;
    float step_declination, step_ascension;
    while (dist_moved + dist_todo >= 1.0) { // there are dabs pending
      { // linear interpolation (nonlinear variant was too slow, see SVN log)
        float frac; // fraction of the remaining distance to move
        if (dist_moved > 0) {
          // "move" the brush exactly to the first dab (moving less than one dab)
          frac = (1.0 - dist_moved) / dist_todo;
          dist_moved = 0;
        } else {
          // "move" the brush from one dab to the next
          frac = 1.0 / dist_todo;
        }
        step_dx        = frac * (x - self->states[MYPAINT_BRUSH_STATE_X]);
        step_dy        = frac * (y - self->states[MYPAINT_BRUSH_STATE_Y]);
        step_dpressure = frac * (pressure - self->states[MYPAINT_BRUSH_STATE_PRESSURE]);
        step_dtime     = frac * (dtime_left - 0.0);
        // Though it looks different, time is interpolated exactly like x/y/pressure.
        step_declination = frac * (tilt_declination - self->states[MYPAINT_BRUSH_STATE_DECLINATION]);
        step_ascension   = frac * smallest_angular_difference(self->states[MYPAINT_BRUSH_STATE_ASCENSION], tilt_ascension);
      }

      update_states_and_setting_values (self, step_dx, step_dy, step_dpressure, step_declination, step_ascension, step_dtime);
      gboolean painted_now = prepare_and_draw_dab (self, surface);
      if (painted_now) {
        painted = YES;
      } else if (painted == UNKNOWN) {
        painted = NO;
      }

      dtime_left   -= step_dtime;
      dist_todo  = count_dabs_to (self, x, y, pressure, dtime_left);
    }

    {
      // "move" the brush to the current time (no more dab will happen)
      // Important to do this at least once every event, because
      // brush_count_dabs_to depends on the radius and the radius can
      // depend on something that changes much faster than only every
      // dab (eg speed).

      step_dx        = x - self->states[MYPAINT_BRUSH_STATE_X];
      step_dy        = y - self->states[MYPAINT_BRUSH_STATE_Y];
      step_dpressure = pressure - self->states[MYPAINT_BRUSH_STATE_PRESSURE];
      step_declination = tilt_declination - self->states[MYPAINT_BRUSH_STATE_DECLINATION];
      step_ascension = smallest_angular_difference(self->states[MYPAINT_BRUSH_STATE_ASCENSION], tilt_ascension);
      step_dtime     = dtime_left;

      //dtime_left = 0; but that value is not used any more

      update_states_and_setting_values (self, step_dx, step_dy, step_dpressure, step_declination, step_ascension, step_dtime);
    }

    // save the fraction of a dab that is already done now
    self->states[MYPAINT_BRUSH_STATE_DIST] = dist_moved + dist_todo;
    //g_print("dist_final = %f\n", states[MYPAINT_BRUSH_STATE_DIST]);

    /* not working any more with the new rng...
    // next seed for the RNG (GRand has no get_state() and states[] must always contain our full state)
    self->states[MYPAINT_BRUSH_STATE_RNG_SEED] = rng_double_next(self->rng);
    */

    // stroke separation logic (for undo/redo)

    if (painted == UNKNOWN) {
      if (self->stroke_current_idling_time > 0 || self->stroke_total_painting_time == 0) {
        // still idling
        painted = NO;
      } else {
        // probably still painting (we get more events than brushdabs)
        painted = YES;
        //if (pressure == 0) g_print ("info: assuming 'still painting' while there is no pressure\n");
      }
    }
    if (painted == YES) {
      //if (stroke_current_idling_time > 0) g_print ("idling ==> painting\n");
      self->stroke_total_painting_time += dtime;
      self->stroke_current_idling_time = 0;
      // force a stroke split after some time
      if (self->stroke_total_painting_time > 4 + 3*pressure) {
        // but only if pressure is not being released
        // FIXME: use some smoothed state for dpressure, not the output of the interpolation code
        //        (which might easily wrongly give dpressure == 0)
        if (step_dpressure >= 0) {
          return TRUE;
        }
      }
    } else if (painted == NO) {
      //if (stroke_current_idling_time == 0) g_print ("painting ==> idling\n");
      self->stroke_current_idling_time += dtime;
      if (self->stroke_total_painting_time == 0) {
        // not yet painted, start a new stroke if we have accumulated a lot of irrelevant motion events
        if (self->stroke_current_idling_time > 1.0) {
          return TRUE;
        }
      } else {
        // Usually we have pressure==0 here. But some brushes can paint
        // nothing at full pressure (eg gappy lines, or a stroke that
        // fades out). In either case this is the prefered moment to split.
        if (self->stroke_total_painting_time+self->stroke_current_idling_time > 0.9 + 5*pressure) {
          return TRUE;
        }
      }
    }
    return FALSE;
  }

#ifdef HAVE_JSON_C
gboolean
update_settings_from_json_object(MyPaintBrush *self)
{
    // Check version
    json_object *version_object = json_object_object_get(self->brush_json, "version");
    int version = json_object_get_int(version_object);
    if (version != 3) {
        fprintf(stderr, "Error: Unsupported brush setting version: %d\n", version);
        return FALSE;
    }

    // Set settings
    json_object *settings = json_object_object_get(self->brush_json, "settings");

    json_object_object_foreach(settings, setting_name, setting_obj) {

        MyPaintBrushSetting setting_id = mypaint_brush_setting_from_cname(setting_name);

        if (!json_object_is_type(setting_obj, json_type_object)) {
            fprintf(stderr, "Error: Wrong type for setting: %s\n", setting_name);
            return FALSE;
        }

        // Base value
        json_object *base_value_obj = json_object_object_get(setting_obj, "base_value");
        double base_value = json_object_get_double(base_value_obj);
        mypaint_brush_set_base_value(self, setting_id, base_value);

        // Inputs
        json_object *inputs = json_object_object_get(setting_obj, "inputs");
        json_object_object_foreach(inputs, input_name, input_obj) {
            MyPaintBrushInput input_id = mypaint_brush_input_from_cname(input_name);

            if (!json_object_is_type(input_obj, json_type_array)) {
                fprintf(stderr, "Error: Wrong inputs type for setting: %s\n", setting_name);
                return FALSE;
            }

            const int number_of_mapping_points = json_object_array_length(input_obj);

            mypaint_brush_set_mapping_n(self, setting_id, input_id, number_of_mapping_points);

            for (int i=0; i<number_of_mapping_points; i++) {
                json_object *mapping_point = json_object_array_get_idx(input_obj, i);

                json_object *x_obj = json_object_array_get_idx(mapping_point, 0);
                float x = json_object_get_double(x_obj);
                json_object *y_obj = json_object_array_get_idx(mapping_point, 1);
                float y = json_object_get_double(y_obj);

                mypaint_brush_set_mapping_point(self, setting_id, input_id, i, x, y);
            }
        }

    }
    return TRUE;
}
#endif // HAVE_JSON_C

gboolean
mypaint_brush_from_string(MyPaintBrush *self, const char *string)
{
#ifdef HAVE_JSON_C
    if (self->brush_json) {
        // Free
        json_object_put(self->brush_json);
    }
    self->brush_json = json_tokener_parse(string);
    return update_settings_from_json_object(self);
#else
    return FALSE;
#endif
}
