#include "neural.h"
#include <stdio.h>
#include <math.h>
/* from math.h */
#define M_LN2		0.69314718055994530942	/* log_e 2 */

#define n_avg 5
double avg_x[n_avg];
double avg_y[n_avg];
double avg_pressure[n_avg];
double avg_speed[n_avg];

/* time in seconds it takes to half the (old) average value */
/*      T0 0 */
#define T1 0.5
#define T2 3.0
#define T3 10.0
#define T4 60.0

void 
update_avg (double * avg, double dt, double value) 
{
  double prev;
  double oldweight;
  prev = avg[0];
  avg[0] = value;

  /* exact: oldweight = exp(-dt*ln(2.0)/T); */
  /* approximation (valid for dt much smaller than T) */
#define UPDATE(i) \
  /*oldweight = 1 + (-dt*M_LN2/T##i); */\
  oldweight = exp(-dt*M_LN2/T##i); \
  avg[i] = oldweight * avg[i] + (1-oldweight)*prev;

  /* manually unrolled loop so there is no division needed once compiled */
  UPDATE(1);
  UPDATE(2);
  UPDATE(3);
  UPDATE(4);
}

double speed_measure_time;
double old_x, old_y;

void 
neural_process_movement (double dt, double x, double y, double z, double d_dist)
{
  if (dt <= 0) return;
  update_avg (avg_x, dt, x);
  update_avg (avg_y, dt, y);
  update_avg (avg_pressure, dt, z);
  /* interesting mistake: update_avg (avg_speed, d_dist, z); */
  printf ("avg_x = %3.3f %3.3f %3.3f %3.3f %3.3f\n", avg_x[4], avg_x[3], avg_x[2], avg_x[1], avg_x[0]);

  speed_measure_time += dt;
  if (speed_measure_time > 0.1) {
    double dx, dy;
    dx = avg_x[1] - old_x; old_x = avg_x[1];
    dy = avg_y[1] - old_y; old_y = avg_y[1];
    printf ("avg_x/y = %3.3f %3.3f\n", avg_x[1], avg_y[1]);
    d_dist = sqrt(dx*dx + dy*dy);
    update_avg (avg_speed, speed_measure_time, (d_dist/speed_measure_time)*avg_pressure[1]);
    /*printf ("avg_speed = %3.3f %3.3f %3.3f %3.3f %3.3f\n", avg_speed[4], avg_speed[3], avg_speed[2], avg_speed[1], avg_speed[0]);*/
    speed_measure_time = 0;
  }
}

void
neural_notice_clear_image (void)
{
}
