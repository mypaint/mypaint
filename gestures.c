/* This file is part of MyPaint.
 * Copyright (C) 2007 by Martin Renold <martinxyz@gmx.ch>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License.
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY. See the COPYING file for more details.
 */

#include "gestures.h"
#include "lfd.h"
#include <stdio.h>
#include <math.h>
#include <assert.h>

int detect_gestures (double dtime, double x, double y, double pressure)
{
  static double time = 0.0;
  time += dtime;
  
  static lfd_t vx, vy, vtot, slow_v;
  vx.time_constant = 0.05;
  vy.time_constant = 0.05;
  vtot.time_constant = 0.1;
  slow_v.time_constant = 5.0;
  
  lfd_update(&vx, dtime, x);
  lfd_update(&vy, dtime, y);
  double v = hypot(vx.filtered_derivative, vy.filtered_derivative);
  lfd_update(&vtot, dtime, v);

  if (pressure > 0 && dtime < 0.1) {
    lfd_update(&slow_v, dtime, v);
  }
  //v = vtot.filtered_signal;

  double v_ratio = v/(slow_v.filtered_signal+0.1);

  static int state = 0;
  static double pressed_time;
  static double released_time;
  static double v_ratio_hist[30];
  static int v_ratio_hist_n;
  double signal = 0;
  if (state == 0) {
    if (pressure > 0) {
      state = 1;
      pressed_time = time;
      v_ratio_hist[0] = v_ratio;
      v_ratio_hist_n = 1;
    }
  } else if (state == 1) {
    double duration = time - pressed_time;
    int n = (int)(duration / 0.01);
    while (n > v_ratio_hist_n-1 && v_ratio_hist_n < 30) {
      v_ratio_hist[v_ratio_hist_n++] = v_ratio;
    }
    if (pressure == 0) {
        released_time = time;
        // experimentally found: v_ratio at 2/3 is best
        double ratio = v_ratio_hist[v_ratio_hist_n*2/3];
        static FILE * hitlog;
        if (!hitlog) hitlog = fopen("hit.log", "a");
        fprintf(hitlog, "%f %d %f %f %f %f\n", ratio, v_ratio_hist_n, duration, v, v_ratio, v_ratio_hist[0]);
        if (ratio < 45.0/100.0) {
          signal = 1;
          //printf("\7");
          //fflush(stdout);
        } else {
          printf("not enough\n");
        }
        state = 0;
    } else if (duration > 0.3) {
      state = 99;
    }
  } else {
    assert(state == 99);
    if (pressure == 0) state = 0;
  }

  static FILE * logfile;
  if (!logfile) logfile = fopen("data.log", "w");
  fprintf(logfile, "%f %f %f %f %f %f %f %f %f\n", time, x, y, pressure, v, vtot.filtered_signal, signal, slow_v.filtered_signal, v_ratio);

  if (signal == 1) {
    return GESTURE_COLORSELECT;
  } else {
    return GESTURE_NONE;
  }
}
