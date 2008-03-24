const int colorselector_size = 256;

#ifndef SWIG

struct PrecalcData {
  int h;
  int s;
  int v;
  //signed char s;
  //signed char v;
};

PrecalcData * precalcData[4];
int precalcDataIndex;

PrecalcData * precalc_data(float phase0)
{
  // Hint to the casual reader: some of the calculation here do not
  // what I originally intended. Not everything here will make sense.
  // It does not matter in the end, as long as the result looks good.

  int width, height;
  float width_inv, height_inv;
  int x, y, i;
  PrecalcData * result;

  width = colorselector_size;
  height = colorselector_size;
  result = (PrecalcData*)g_malloc(sizeof(PrecalcData)*width*height);

  //phase0 = rand_double (rng) * 2*M_PI;

  width_inv = 1.0/width;
  height_inv = 1.0/height;

  i = 0;
  for (y=0; y<height; y++) {
    for (x=0; x<width; x++) {
      float h, s, v, s_original, v_original;
      int dx, dy;
      float v_factor = 0.8;
      float s_factor = 0.8;
      float h_factor = 0.05;

#define factor2_func(x) ((x)*(x)*SIGN(x))
      float v_factor2 = 0.01;
      float s_factor2 = 0.01;


      h = 0;
      s = 0;
      v = 0;

      dx = x-width/2;
      dy = y-height/2;

      // basically, its x-axis = value, y-axis = saturation
      v = dx*v_factor + factor2_func(dx)*v_factor2;
      s = dy*s_factor + factor2_func(dy)*s_factor2;

      v_original = v; s_original = s;

      // overlay sine waves to color hue, not visible at center, ampilfying near the border
      if (1) {
        float amplitude, phase;
        float dist, dist2, borderdist;
        float dx_norm, dy_norm;
        float angle;
        dx_norm = dx*width_inv;
        dy_norm = dy*height_inv;

        dist2 = dx_norm*dx_norm + dy_norm*dy_norm;
        dist = sqrtf(dist2);
        borderdist = 0.5 - MAX(ABS(dx_norm), ABS(dy_norm));
        angle = atan2f(dy_norm, dx_norm);
        amplitude = 50 + dist2*dist2*dist2*100;
        phase = phase0 + 2*M_PI* (dist*0 + dx_norm*dx_norm*dy_norm*dy_norm*50) + angle*7;
        //h = sinf(phase) * amplitude;
        h = sinf(phase);
        h = (h>0)?h*h:-h*h;
        h *= amplitude;

        // calcualte angle to next 45-degree-line
        angle = ABS(angle)/M_PI;
        if (angle > 0.5) angle -= 0.5;
        angle -= 0.25;
        angle = ABS(angle) * 4;
        // angle is now in range 0..1
        // 0 = on a 45 degree line, 1 = on a horizontal or vertical line

        v = 0.6*v*angle + 0.4*v;
        h = h * angle * 1.5;
        s = s * angle * 1.0;

        // this part is for strong color variations at the borders
        if (borderdist < 0.3) {
          float fac;
          float h_new;
          fac = (1 - borderdist/0.3);
          // fac is 1 at the outermost pixels
          v = (1-fac)*v + fac*0;
          s = (1-fac)*s + fac*0;
          fac = fac*fac*0.6;
          h_new = (angle+phase0+M_PI/4)*360/(2*M_PI) * 8;
          while (h_new > h + 360/2) h_new -= 360;
          while (h_new < h - 360/2) h_new += 360;
          h = (1-fac)*h + fac*h_new;
          //h = (angle+M_PI/4)*360/(2*M_PI) * 4;
        }
      }

      {
        // undo that funky stuff on horizontal and vertical lines
        int min = ABS(dx);
        if (ABS(dy) < min) min = ABS(dy);
        if (min < 30) {
          float mul;
          min -= 6;
          if (min < 0) min = 0;
          mul = min / (30.0-1.0-6.0);
          h = mul*h; //+ (1-mul)*0;

          v = mul*v + (1-mul)*v_original;
          s = mul*s + (1-mul)*s_original;
        }
      }

      h -= h*h_factor;

      result[i].h = (int)h;
      result[i].v = (int)v;
      result[i].s = (int)s;
      i++;
    }
  }
  return result;
}

#endif /* #ifndef SWIG */

void render_colorselector(PyObject * arr, double color_h, double color_s, double color_v)
{
  PrecalcData * pre;
  guchar * pixels;
  int x, y;
  int h, s, v;
  int base_h, base_s, base_v;

  assert(ISCARRAY(arr));
  assert(ISBEHAVED(arr));
  assert(PyArray_DIMS(arr) == 3);
  assert(PyArray_DIM(arr, 0) == colorselector_size);
  assert(PyArray_DIM(arr, 1) == colorselector_size);
  assert(PyArray_DIM(arr, 2) == 3);
  pixels = (guchar*)((PyArrayObject*)arr)->data;
    
  pre = precalcData[precalcDataIndex];
  if (!pre) {
    pre = precalcData[precalcDataIndex] = precalc_data(2*M_PI*(precalcDataIndex/4.0));
  }
  precalcDataIndex++;
  precalcDataIndex %= 4;

  base_h = color_h*360;
  base_s = color_s*255;
  base_v = color_v*255;

  for (y=0; y<colorselector_size; y++) {
    for (x=0; x<colorselector_size; x++) {
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

      p = pixels + 3*(y*colorselector_size + x);
      hsv_to_rgb_int (&h, &s, &v);
      p[0] = h; p[1] = s; p[2] = v;
    }
  }
}

