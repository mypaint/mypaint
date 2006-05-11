#include <math.h>
#include "brush_dab.h"
// This actually draws every pixel of the dab.
// Called from brush_prepare_and_draw_dab.
// The parameters are only in the header file to avoid duplication.
{
  float r_fringe;
  int x0, y0;
  int x1, y1;
  int xp, yp;
  guchar *rgb;
  float xx, yy, rr;
  float radius2, one_over_radius2;
  //float precalc1, precalc2;
  int a;
  //int m1, m2; 
  guchar randoms[8];
  guchar random_pos;

  guchar c[3];
  if (!s) return;
  
  g_assert (hardness <= 1.0 && hardness >= 0.0);
  if (hardness == 0) return; // infintly small point, rest transparent

  r_fringe = radius + 1;
  x0 = floor (x - r_fringe);
  y0 = floor (y - r_fringe);
  x1 = ceil (x + r_fringe);
  y1 = ceil (y + r_fringe);
  if (x0 < 0) x0 = 0;
  if (y0 < 0) y0 = 0;
  if (x1 > s->w) x1 = s->w;
  if (y1 > s->h) y1 = s->h;
  rr = sqr(radius);
  if (radius < 0.1) return;
  c[0] = color[0];
  c[1] = color[1];
  c[2] = color[2];
  radius2 = sqr(radius);
  one_over_radius2 = 1.0/radius2;
  
  // precalculate randomness
  ((guint32*)randoms)[0] = g_random_int ();
  ((guint32*)randoms)[1] = g_random_int ();
  random_pos = 0;

  g_assert (opaque >= 0 && opaque <= 1);
  if (opaque == 0) return;

  a = floor(saturation_slowdown * 256 + 0.5);
  if (a < -256) a = -256; if (a > 256) a = 256;

  for (yp = y0; yp < y1; yp++) {
    yy = (yp + 0.5 - y);
    yy *= yy;
    for (xp = x0; xp < x1; xp++) {
      xx = (xp + 0.5 - x);
      xx *= xx;
      rr = (yy + xx) * one_over_radius2;
      // rr is in range 0.0..1.0*sqrt(2)
      rgb = PixelXY(s, xp, yp);

      if (rr <= 1.0) {
        int opa;
        { // hardness
          float o;
          if (hardness == 1) {
            o = 1.0;
          } else if (rr < hardness) {
            o = rr + 1-(rr/hardness);
            // hardness == 0 is nonsense, excluded above
          } else {
            o = hardness/(hardness-1)*(rr-1);
            // hardness == 1 ?
          }
          opa = floor(opaque * o * 256 + 0.5);
          g_assert (opa >= 0 && opa <= 256);
          // opa is in range 0..256
        }
        
        int rgbdiff[3];
        int diff_sum;
        diff_sum = 0;
        rgbdiff[0] = c[0] - rgb[0]; diff_sum += ABS(rgbdiff[0]);
        rgbdiff[1] = c[1] - rgb[1]; diff_sum += ABS(rgbdiff[1]);
        rgbdiff[2] = c[2] - rgb[2]; diff_sum += ABS(rgbdiff[2]);
        // rgbdiff[] is in range -255..+255
        // dif_sum is in range 0..3*255

        if (saturation_slowdown) {
          // FIXME: still buggy at high diff_sum, even if saturation_slowdown == 0
          int a_times_diff_sum;
          a_times_diff_sum = a*diff_sum / 256;
          // a_times_diff_sum is in range 0..3*255/256

          // Formula: o = o * ( a*d*d + d - a*d )  all ranges 0..1, d=diff_sum
          opa = opa * (a_times_diff_sum*diff_sum + diff_sum*3*255 - a_times_diff_sum*3*255);
          // opa is in range 0..256*3*3*255*255, which is way too much precision
          opa /= 3*3*255*255;
          //if (opa == 0) opa = 1; // give it a chance to make the final step
          // opa is in range 1..256
        }

        rgbdiff[0] *= opa;
        rgbdiff[1] *= opa;
        rgbdiff[2] *= opa;
        // rgbdiff has range -255*256..+255*256

        int i;
        for (i=0; i<3; i++) {
          int reminder;
          int negative;
          if (rgbdiff[i] < 0) {
            negative = 1;
            rgbdiff[i] = - rgbdiff[i];
          } else {
            negative = 0;
          }
          reminder = rgbdiff[i] % 256;
          rgbdiff[i] /= 256;
          // use randomness to fake more precision
          // - ah, I just learned that this is called "dither". I hope I've done it right.
          random_pos = (random_pos + 1 + rgbdiff[i] % 1 /* hope that's slightly random */) % 8;
          if (reminder > randoms[random_pos] /* 0..255 */) {
            rgbdiff[i]++;
          }
          if (negative) rgbdiff[i] = - rgbdiff[i];
        }
        
        rgb[0] += rgbdiff[0];
        rgb[1] += rgbdiff[1];
        rgb[2] += rgbdiff[2];
      }
      rgb += 3;
    }
  }
  
  if (bbox) {
    // expand the bounding box to include the region we just drawed
    int bb_x, bb_y, bb_w, bb_h;
    bb_x = floor(x - (radius+1));
    bb_y = floor(y - (radius+1));
    /* FIXME: think about it exactly */
    bb_w = ceil (2*(radius+1));
    bb_h = ceil (2*(radius+1));

    ExpandRectToIncludePoint(bbox, bb_x, bb_y);
    ExpandRectToIncludePoint(bbox, bb_x+bb_w-1, bb_y+bb_h-1);
  }
}

