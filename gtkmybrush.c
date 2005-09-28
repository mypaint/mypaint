// gtk stock code - left gtk prefix to use the pygtk wrapper-generator easier
#include <stdio.h>
#include <string.h>
#include <glib.h>
#include <math.h>
#include "gtkmybrush.h"
#include "helpers.h"
#include "brush_dab.h" 
;  // ; needed

#define DEBUGLOG 0
#define LINEAR_INTERPOLATION 1

#define abs(x) (((x)>0)?(x):(-(x)))

void
gtk_my_brush_set_base_value (GtkMyBrush * b, int id, float value)
{
  g_assert (id >= 0 && id < BRUSH_SETTINGS_COUNT);
  b->settings[id].base_value = value;
}

void gtk_my_brush_set_mapping (GtkMyBrush * b, int id, int input, int index, float value)
{
  int i;
  g_assert (id >= 0 && id < BRUSH_SETTINGS_COUNT);
  g_assert (input >= 0 && input < INPUT_COUNT);
  g_assert (index >= 0 && index < 8);

  //g_print("set mapping: id=%d, input=%d, index=%d, value=%f\n", id, input, index, value);
  Mapping * m = b->settings[id].mapping[input];
  if (!m) {
    m = b->settings[id].mapping[input] = g_malloc(sizeof(Mapping));
    for (i=0; i<4; i++) {
      m->xvalues[i] = 0;
      m->yvalues[i] = 0;
    }
  }
  if (index % 2 == 0) {
    m->xvalues[index/2] = value;
  } else {
    m->yvalues[index/2] = value;
  }
}

void gtk_my_brush_remove_mapping (GtkMyBrush * b, int id, int input)
{
  g_assert (id >= 0 && id < BRUSH_SETTINGS_COUNT);
  g_assert (input >= 0 && input < INPUT_COUNT);

  //g_print("remove mapping: id=%d, input=%d\n", id, input);

  g_free(b->settings[id].mapping[input]);
  b->settings[id].mapping[input] = NULL;
}

void
gtk_my_brush_set_color (GtkMyBrush * b, int red, int green, int blue)
{
  g_assert (red >= 0 && red <= 255);
  g_assert (green >= 0 && green <= 255);
  g_assert (blue >= 0 && blue <= 255);
  b->color[0] = red;
  b->color[1] = green;
  b->color[2] = blue;
}

void
gtk_my_brush_set_print_inputs (GtkMyBrush * b, int value)
{
  b->print_inputs = value;
}

float
gtk_my_brush_get_painting_time (GtkMyBrush * b)
{
  return b->painting_time;
}

void
gtk_my_brush_set_painting_time (GtkMyBrush * b, float value)
{
  b->painting_time = value;
}

static void gtk_my_brush_class_init    (GtkMyBrushClass *klass);
static void gtk_my_brush_init          (GtkMyBrush      *b);
static void gtk_my_brush_finalize (GObject *object);

static gpointer parent_class;

GType
gtk_my_brush_get_type (void)
{
  static GType my_brush_type = 0;

  if (!my_brush_type)
    {
      static const GTypeInfo my_brush_info =
      {
	sizeof (GtkMyBrushClass),
	NULL,		/* base_init */
	NULL,		/* base_finalize */
	(GClassInitFunc) gtk_my_brush_class_init,
	NULL,		/* class_finalize */
	NULL,		/* class_data */
	sizeof (GtkMyBrush),
	0,		/* n_preallocs */
	(GInstanceInitFunc) gtk_my_brush_init,
      };

      my_brush_type =
	g_type_register_static (G_TYPE_OBJECT, "GtkMyBrush",
				&my_brush_info, 0);
    }

  return my_brush_type;
}

static void
gtk_my_brush_class_init (GtkMyBrushClass *class)
{
  GObjectClass *gobject_class = G_OBJECT_CLASS (class);
  parent_class = g_type_class_peek_parent (class);
  gobject_class->finalize = gtk_my_brush_finalize;
}

static void
gtk_my_brush_init (GtkMyBrush *b)
{
  int i, j;
  for (i=0; i<BRUSH_SETTINGS_COUNT; i++) {
    for (j=0; j<INPUT_COUNT; j++) {
      b->settings[i].mapping[j] = NULL;
    }
  }
  b->painting_time = 0;
  // defaults will be set from python
}

static void
gtk_my_brush_finalize (GObject *object)
{
  GtkMyBrush * b;
  int i, j;
  g_return_if_fail (object != NULL);
  g_return_if_fail (GTK_IS_MY_BRUSH (object));
  b = GTK_MY_BRUSH (object);
  for (i=0; i<BRUSH_SETTINGS_COUNT; i++) {
    for (j=0; j<INPUT_COUNT; j++) {
      g_free(b->settings[i].mapping[j]);
    }
  }
  G_OBJECT_CLASS (parent_class)->finalize (object);
}

GtkMyBrush*
gtk_my_brush_new (void)
{
  g_print ("This gets never called... but is needed. Strange.\n");
  return g_object_new (GTK_TYPE_MY_BRUSH, NULL);
}

// returns the fraction still left after t seconds
float exp_decay (float T_const, float t)
{
  // FIXME: think about whether the argument make mathematical sense
  if (T_const <= 0.001) {
    return 0.0;
  } else {
    return exp(- t / T_const);
  }
}

void brush_reset (GtkMyBrush * b)
{
  b->time = 0; // triggers the real reset below in brush_stroke_to
}

// high-level part of before each dab
void brush_prepare_and_draw_dab (GtkMyBrush * b, Surface * s, Rect * bbox)
{
  float x, y, radius_log, opaque;
  int i, j;
  gint color[3];
  int color_is_hsv;
  float pressure;
  float settings[BRUSH_SETTINGS_COUNT];
  float inputs[INPUT_COUNT];
  float base_radius_pixels = expf(b->settings[BRUSH_RADIUS_LOGARITHMIC].base_value);

  // FIXME: does happen (interpolation problem?)
  if (b->pressure < 0.0) b->pressure = 0.0;
  if (b->pressure > 1.0) b->pressure = 1.0;
  g_assert (b->pressure >= 0.0 && b->pressure <= 1.0);
  pressure = b->pressure; // could distort it here

  float norm_dx, norm_dy, norm_dist, norm_speed;
  norm_dx = b->dx / b->dtime / base_radius_pixels;
  norm_dy = b->dy / b->dtime / base_radius_pixels;
  norm_speed = sqrt(sqr(norm_dx) + sqr(norm_dy));
  norm_dist = norm_speed * b->dtime;

  inputs[INPUT_PRESSURE] = pressure;
  inputs[INPUT_SPEED]  = b->norm_speed_slow1 * 0.002;
  inputs[INPUT_SPEED2] = b->norm_speed_slow2 * 0.005;
  inputs[INPUT_RANDOM] = 0.5; // actually unused
  inputs[INPUT_STROKE_LENGTH] = b->stroke_length * 0.05;
  if (b->print_inputs) {
    g_print("press=% 4.3f, speed=% 4.4f\tspeed2=% 4.4f\n", inputs[INPUT_PRESSURE], inputs[INPUT_SPEED], inputs[INPUT_SPEED2]);
  }

  for (i=0; i<BRUSH_SETTINGS_COUNT; i++) {
    settings[i] = b->settings[i].base_value;

    // FIXME: each setting might depend on any input however, some
    // inputs need the value of a setting to be updated. Using the old
    // value here, this results in a small lag that would be avoidable
    // in most cases. -- however if one of the inputs jumps between
    // two strokes (speed for example) this might be fatal.

    // ==> the inputs should be updated above, and the settings needed
    // to do so should have been cached from the last step...

    for (j=0; j<INPUT_COUNT; j++) {
      Mapping * m = b->settings[i].mapping[j];
      // OPTIMIZE?
      if (m) {
        float x, y;
        if (j == INPUT_RANDOM) {
          x = g_random_double ();
        } else {
          x = inputs[j];
        }
        //if (i == 2 && j == 3) g_print("x = %f ", x);
        int p0;
        float x0, y0, x1, y1;
        // decide what region to use
        p0 = -1; // left point of the linear region (-1 is the implicit x=0,y=0 point)
        while (p0+1 < 4 // not in the last region already
               && x > m->xvalues[p0+1] // x position is further right than the current region
               && m->xvalues[p0+2] > 0 // next enpoint does exists (points with x=0 are disabled)
               ) p0++;
        x0 = (p0 == -1) ? 0 : m->xvalues[p0];
        y0 = (p0 == -1) ? 0 : m->yvalues[p0];
        x1 = m->xvalues[p0+1];
        y1 = m->yvalues[p0+1];
        // linear interpolation
        float m, q;
        m = (y1-y0)/(x1-x0);
        q = y0 - m*x0;
        y = m*x + q;
        settings[i] += y;
        //if (i == 2 && j == 3) g_print("y = %f (p0=%d, %f %f %f %f)\n", y, p0, x0, y0, x1, y1);
      }
    }
  }
    

  {
    float fac = 1.0 - exp_decay (settings[BRUSH_SLOW_TRACKING_PER_DAB], 1.0);
    b->x_slow += (b->x - b->x_slow) * fac; // FIXME: should depend on base radius?
    b->y_slow += (b->y - b->y_slow) * fac;
    x = b->x_slow;
    y = b->y_slow;
  }

  radius_log = settings[BRUSH_RADIUS_LOGARITHMIC];
  opaque = settings[BRUSH_OPAQUE] * settings[BRUSH_OPAQUE_MULTIPLY];
  if (opaque >= 1.0) opaque = 1.0;
  if (opaque <= 0.0) opaque = 0.0;

  { // slow speed
    float fac;
    fac = 1.0 - exp_decay (settings[BRUSH_SPEED1_SLOWNESS], b->dtime);
    b->norm_speed_slow1 += (norm_speed - b->norm_speed_slow1) * fac;
    fac = 1.0 - exp_decay (settings[BRUSH_SPEED2_SLOWNESS], b->dtime);
    b->norm_speed_slow2 += (norm_speed - b->norm_speed_slow2) * fac;
  }
  
  { // slow speed, but as vector this time
    float fac = 1.0 - exp_decay (exp(settings[BRUSH_OFFSET_BY_SPEED_SLOWNESS]*0.01)-1.0, b->dtime);
    b->norm_dx_slow += (norm_dx - b->norm_dx_slow) * fac;
    b->norm_dy_slow += (norm_dy - b->norm_dy_slow) * fac;
  }

  { // stroke length
    b->stroke_length += norm_dist;
    if (!b->stroke_started) {
      if (b->pressure > settings[BRUSH_STROKE_TRESHOLD]) {
        // start new stroke
        b->stroke_started = 1;
        b->stroke_length = 0.0;
      }
    } else {
      if (b->pressure <= settings[BRUSH_STROKE_TRESHOLD] * 0.9) {
        // end stroke
        b->stroke_started = 0;
      }
    }
  }

  if (DEBUGLOG) {
    static FILE * logfile = NULL;
    if (!logfile) {
      logfile = fopen("dabinput.log", "w");
    }
    fprintf(logfile, "%f %f %f %f %f\n", b->time, b->dtime, b->x, b->dx, b->norm_dx_slow);
  }

  if (settings[BRUSH_OFFSET_BY_SPEED]) {
    x += b->norm_dx_slow * settings[BRUSH_OFFSET_BY_SPEED] * 0.1 * base_radius_pixels;
    y += b->norm_dy_slow * settings[BRUSH_OFFSET_BY_SPEED] * 0.1 * base_radius_pixels;
  }

  if (settings[BRUSH_OFFSET_BY_RANDOM]) {
    x += gauss_noise () * settings[BRUSH_OFFSET_BY_RANDOM] * base_radius_pixels;
    y += gauss_noise () * settings[BRUSH_OFFSET_BY_RANDOM] * base_radius_pixels;
  }

  // change base radius
  b->settings[BRUSH_RADIUS_LOGARITHMIC].base_value += settings[BRUSH_CHANGE_RADIUS] * 0.01;

  // color part
  
  for (i=0; i<3; i++) color[i] = b->color[i];
  color_is_hsv = 0;

  if (settings[BRUSH_ADAPT_COLOR_FROM_IMAGE]) {
    int px, py;
    guchar *rgb;
    float v = settings[BRUSH_ADAPT_COLOR_FROM_IMAGE];
    px = ROUND(x);
    py = ROUND(y);
    if (px < 0) px = 0;
    if (py < 0) py = 0;
    if (px > s->w-1) px = s->w - 1;
    if (py > s->h-1) py = s->h - 1;
    rgb = PixelXY(s, px, py);
    for (i=0; i<3; i++) {
      color[i] = ROUND((1.0-v)*color[i] + v*rgb[i]);
      if (color[i] < 0) color[i] = 0;
      if (color[i] > 255) color[i] = 255;
      b->color[i] = color[i];
    }
  }

  if (settings[BRUSH_COLOR_VALUE] ||
      settings[BRUSH_COLOR_SATURATION] ||
      settings[BRUSH_COLOR_HUE]) {
    color_is_hsv = 1;
    gimp_rgb_to_hsv_int (color + 0, color + 1, color + 2);
  }
  

  if (settings[BRUSH_COLOR_HUE]) {
    g_assert (color_is_hsv);
    color[0] += ROUND (settings[BRUSH_COLOR_HUE] * 64.0);
  }
  if (settings[BRUSH_COLOR_SATURATION]) {
    g_assert (color_is_hsv);
    color[1] += ROUND (settings[BRUSH_COLOR_SATURATION] * 128.0);
  }
  if (settings[BRUSH_COLOR_VALUE]) {
    g_assert (color_is_hsv);
    color[2] += ROUND (settings[BRUSH_COLOR_VALUE] * 128.0);
  }

  { // final calculations
    float radius;
    guchar c[3];
    radius = expf(radius_log);
    
    if (radius < 0) radius = 0; // don't ask me why this happens
    //FIXME: performance problem acutally depending on CPU
    if (radius > 100) radius = 100;
    g_assert(opaque >= 0);
    g_assert(opaque <= 1);
    
    // used for interpolation later
    b->actual_radius = radius < 0.1 ? 0.1 : radius;
    
    if (color_is_hsv) {
      while (color[0] < 0) color[0] += 360;
      while (color[0] > 360) color[0] -= 360;
      if (color[1] < 0) color[1] = 0;
      if (color[1] > 255) color[1] = 255;
      if (color[2] < 0) color[2] = 0;
      if (color[2] > 255) color[2] = 255;
      gimp_hsv_to_rgb_int (color + 0, color + 1, color + 2);
    }
    for (i=0; i<3; i++) c[i] = color[i];

    float hardness = settings[BRUSH_HARDNESS];
    if (hardness > 1.0) hardness = 1.0;
    if (hardness < 0.0) hardness = 0.0;
    float sat_slowdown = settings[BRUSH_SATURATION_SLOWDOWN];
    if (sat_slowdown > 1.0) sat_slowdown = 1.0;
    if (sat_slowdown < 0.0) sat_slowdown = 0.0;

    draw_brush_dab (s, bbox,
                    x, y, radius, opaque, hardness, c, sat_slowdown);
  }
}

float brush_count_dabs_to (GtkMyBrush * b, float x, float y, float pressure, float time)
{
  float dx, dy, dt;
  float res1, res2, res3;
  float dist;
  if (b->actual_radius == 0.0) b->actual_radius = expf(b->settings[BRUSH_RADIUS_LOGARITHMIC].base_value);
  if (b->actual_radius < 0.5) b->actual_radius = 0.5;
  if (b->actual_radius > 500.0) b->actual_radius = 500.0;
  dx = x - b->x;
  dy = y - b->y;
  //dp = pressure - b->pressure; // Not useful?
  dt = time - b->time;

  dist = sqrtf (dx*dx + dy*dy);
  res1 = dist / b->actual_radius * b->settings[BRUSH_DABS_PER_ACTUAL_RADIUS].base_value /*FIXME?*/;
  res2 = dist / expf(b->settings[BRUSH_RADIUS_LOGARITHMIC].base_value) * b->settings[BRUSH_DABS_PER_BASIC_RADIUS].base_value /*FIXME?*/;
  res3 = dt * b->settings[BRUSH_DABS_PER_SECOND].base_value /*FIXME?*/;
  return res1 + res2 + res3;
}

void brush_stroke_to (GtkMyBrush * b, Surface * s, float x, float y, float pressure, float time, Rect * bbox)
{
  float dist;
  if (DEBUGLOG) { // logfile for debugging
    static FILE * logfile = NULL;
    if (!logfile) {
      logfile = fopen("rawinput.log", "w");
    }
    fprintf(logfile, "%f %f %f %f\n", time, x, y, pressure);
  }
  if (time <= b->time) {
    //g_print("timeskip\n");
    return;
  }

  if (b->time == 0 || time - b->time > 5) {
    // reset
    b->x = x;
    b->y = y;
    b->pressure = pressure;
    b->time = time;

    b->last_time = b->time;
    b->x_slow = b->x;
    b->y_slow = b->y;
    b->norm_dx_slow = 0.0;
    b->norm_dy_slow = 0.0;
    b->stroke_started = 0;
    b->stroke_length = 100.0; // start in a state as if the stroke was long finished
    return;
  }

  if (pressure > 0) {
    b->painting_time += time - b->last_time;
  }

  { // calculate the actual "virtual" cursor position
    float fac = 1.0 - exp_decay (b->settings[BRUSH_SLOW_TRACKING].base_value, 100.0*(time - b->time));
    x = b->x + (x - b->x) * fac;
    y = b->y + (y - b->y) * fac;
  }
  // draw many (or zero) dabs to the next position
  dist = brush_count_dabs_to (b, x, y, pressure, time);
  if (dist > 300) {
    //g_print ("Warning: NOT drawing %f dabs, resetting brush instead.\n", dist);
    b->time = 0; // reset
    return;
  }

  //g_print("dist = %f\n", dist);
  // Not going to recalculate dist each step.

  if (LINEAR_INTERPOLATION) {
    while (dist >= 1.0) {
      { // linear interpolation
        // Inside the loop because outside it produces numerical errors
        // resulting in b->pressure being small negative and such effects.
        float step;
        step = 1 / dist;
        b->dx        = step * (x - b->x);
        b->dy        = step * (y - b->y);
        b->dpressure = step * (pressure - b->pressure);
        b->dtime     = step * (time - b->time);
      }
      
      b->x        += b->dx;
      b->y        += b->dy;
      b->pressure += b->dpressure;
      b->time     += b->dtime;
      
      dist -= 1.0;
      
      brush_prepare_and_draw_dab (b, s, bbox);
    }
  } else {
    // cubic interpolation (so b->dx does not jump between dabs)
    // constant accelerations (the "time" variable is the step, here called dist):
    float ax, ay, ap, at;
    float step, stepstep;
    step = 1 / dist;
    stepstep = step*step;
    ax = (x        - b->x)        *stepstep  - b->dx       *step;
    ay = (y        - b->y)        *stepstep  - b->dy       *step;
    ap = (pressure - b->pressure) *stepstep  - b->dpressure*step;
    at = (time     - b->time)     *stepstep  - b->dtime    *step;
    //g_print("%f %f %f %f\n", ax, ay, ap, at);
    while (dist >= 1.0) {
      // This is probably not what my numeric teacher told me about solving
      // differential equations. I can't recall right now, so never mind ;-)
      b->dx        += ax;
      b->dy        += ay;
      b->dpressure += ap;
      b->dtime     += at;
      
      b->x        += b->dx;
      b->y        += b->dy;
      b->pressure += b->dpressure;
      b->time     += b->dtime;
      
      dist -= 1.0;
      
      brush_prepare_and_draw_dab (b, s, bbox);
    }
  }

  // not equal to b_time now unless dist == 0
  b->last_time = time;


  if (pressure == 0) {
    // stroke is certainly finished now (interpolation issue)
    b->stroke_started = 0;
  }
}

#define SIZE 256
typedef struct {
  int h;
  int s;
  int v;
  //signed char s;
  //signed char v;
} PrecalcData;

PrecalcData * precalcData[4];
int precalcDataIndex;

PrecalcData * precalc_data(float phase0)
{
  int width, height;
  float width_inv, height_inv;
  int x, y, i;
  PrecalcData * result;

  width = SIZE;
  height = SIZE;
  result = malloc(sizeof(PrecalcData)*width*height);

  //phase0 = g_random_double () * 2*M_PI;

  width_inv = 1.0/width;
  height_inv = 1.0/height;

  i = 0;
  for (y=0; y<height; y++) {
    for (x=0; x<width; x++) {
      int h, s, v;
      int dx, dy;
      float v_factor = 2.0;
      float s_factor = 2.0;
      
      h = 0;
      s = 0;
      v = 0;

      dx = x-width/2;
      dy = y-height/2;

      // basically, its x-axis = value, y-axis = saturation
      v = dx*v_factor;
      s = dy*s_factor;


      // overlay sine waves to color hue, not visible at center, ampilfying near the border
      if (1) {
        float amplitude, phase;
        float dist, dist2;
        float dx_norm, dy_norm;
        float angle;
        dx_norm = dx*width_inv;
        dy_norm = dy*height_inv;

        dist2 = dx_norm*dx_norm + dy_norm*dy_norm;
        dist = sqrtf(dist2);
        angle = atan2f(dy_norm, dx_norm);
        amplitude = 50 + dist2*dist2*dist2*100;
        phase = phase0 + 2*M_PI* (dist*0 + dx_norm*dx_norm*dy_norm*dy_norm*50) + angle*7;
        h = sinf(phase) * amplitude;

        // calcualte angle to next 45-degree-line
        angle = abs(angle)/M_PI;
        if (angle > 0.5) angle -= 0.5;
        angle -= 0.25;
        angle = abs(angle) * 4;
        // angle is now in range 0..1
        // 0 = on a 45 degree line, 1 = on a horizontal or vertical line

        v = 0.6*v*angle + 0.4*v;
        h = h * angle * 1.5;
        s = s * angle * 1.0;
      }

      {
        // undo that funky stuff on horizontal and vertical lines
        int min = abs(dx);
        if (abs(dy) < min) min = abs(dy);
        if (min < 30) {
          float mul;
          min -= 6;
          if (min < 0) min = 0;
          mul = min / (30.0-1.0-6.0);
          h = mul*h; //+ (1-mul)*0;
          v = mul*v + (1-mul)*dx*v_factor;
          s = mul*s + (1-mul)*dy*s_factor;
        }
      }

      result[i].h = h;
      result[i].v = v;
      result[i].s = s;
      i++;
    }
  }
  return result;
}

GdkPixbuf* gtk_my_brush_get_colorselection_pixbuf (GtkMyBrush * b)
{
  GdkPixbuf* pixbuf;
  PrecalcData * pre;
  guchar * pixels;
  int rowstride, n_channels;
  int x, y;
  int h, s, v;
  int base_h, base_s, base_v;

  pixbuf = gdk_pixbuf_new (GDK_COLORSPACE_RGB, /*has_alpha*/0, /*bits_per_sample*/8, SIZE, SIZE);

  pre = precalcData[precalcDataIndex];
  if (!pre) {
    pre = precalcData[precalcDataIndex] = precalc_data(2*M_PI*(precalcDataIndex/4.0));
  }
  precalcDataIndex++;
  precalcDataIndex %= 4;

  n_channels = gdk_pixbuf_get_n_channels (pixbuf);
  g_assert (!gdk_pixbuf_get_has_alpha (pixbuf));
  g_assert (n_channels == 3);

  rowstride = gdk_pixbuf_get_rowstride (pixbuf);
  pixels = gdk_pixbuf_get_pixels (pixbuf);

  base_h = b->color[0];
  base_s = b->color[1];
  base_v = b->color[2];
  gimp_rgb_to_hsv_int (&base_h, &base_s, &base_v);

  for (y=0; y<SIZE; y++) {
    for (x=0; x<SIZE; x++) {
      guchar * p;

      h = base_h + pre->h;
      s = base_s + pre->s;
      v = base_v + pre->v;
      pre++;


      if (s < 0) { if (s < -50) { s = - (s + 50); } else { s = 0; } } 
      if (s > 255) { if (s > 255 + 50) { s = 255 - ((s-50)-255); } else { s = 255; } }
      if (v < 0) { if (v < -50) { v = - (v + 50); } else { v = 0; } }
      if (v > 255) { if (v > 255 + 50) { v = 255 - ((v-50)-255); } else { v = 255; } }

      s = s & 255;
      v = v & 255;
      h = h%360; if (h<0) h += 360;

      p = pixels + y * rowstride + x * n_channels;
      gimp_hsv_to_rgb_int (&h, &s, &v);
      p[0] = h; p[1] = s; p[2] = v;
    }
  }
  return pixbuf;
}
