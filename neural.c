#include "neural.h"
#include "nntrainer.h"
#include <stdio.h>
#include <glib.h>
#include <math.h>
/* from math.h */
#define M_LN2		0.69314718055994530942	/* log_e 2 */

struct trainer *ann;

#define n_avg 5
float avg_x[n_avg];
float avg_y[n_avg];
float avg_pressure[n_avg];
float avg_speed[n_avg];
float clear_dist;

#define n_inputs 15
#define n_outputs 1

/* time in seconds it takes to half the (old) average value */
/*      T0 0 */
#define T1 0.5
#define T2 3.0
#define T3 10.0
#define T4 60.0

void 
update_avg (float * avg, float dt, float value) 
{
  float prev;
  float oldweight;
  prev = avg[0];
  avg[0] = value;

#define UPDATE(i) \
  oldweight = exp(-dt*M_LN2/T##i); \
  avg[i] = oldweight * avg[i] + (1-oldweight)*prev;

  /* manually unrolled loop so there is no division needed once compiled */
  UPDATE(1);
  UPDATE(2);
  UPDATE(3);
  UPDATE(4);
}

void neural_datapoint (void);

float speed_measure_time;
float nn_datapoint_time;
float old_x, old_y;

void 
neural_process_movement (float dt, float x, float y, float z, float d_dist, int record_data, int guess_size)
{
  if (dt <= 0) return;
  update_avg (avg_x, dt, x);
  update_avg (avg_y, dt, y);
  update_avg (avg_pressure, dt, z);
  /* interesting mistake: update_avg (avg_speed, d_dist, z); */
  /* printf ("avg_x = %3.3f %3.3f %3.3f %3.3f %3.3f\n", avg_x[4], avg_x[3], avg_x[2], avg_x[1], avg_x[0]); */

  speed_measure_time += dt;
  if (speed_measure_time > 0.1) {
    float dx, dy;
    dx = avg_x[1] - old_x; old_x = avg_x[1];
    dy = avg_y[1] - old_y; old_y = avg_y[1];
    /* printf ("avg_x/y = %3.3f %3.3f\n", avg_x[1], avg_y[1]); */
    d_dist = sqrt(dx*dx + dy*dy);
    update_avg (avg_speed, speed_measure_time, (d_dist/speed_measure_time)*avg_pressure[1]);
    /*printf ("avg_speed = %3.3f %3.3f %3.3f %3.3f %3.3f\n", avg_speed[4], avg_speed[3], avg_speed[2], avg_speed[1], avg_speed[0]);*/
    speed_measure_time = 0;
    if (clear_dist > 0) {
      /* forget that the image was cleared as drawing goes on    (T=30.0) */
      clear_dist *= exp(-(speed_measure_time*avg_pressure[1])*M_LN2/30.0);
    }
  }

  if (record_data) {
    nn_datapoint_time += dt;
    if (nn_datapoint_time > 5.0) {
      nn_datapoint_time = 0;
      neural_datapoint ();
    }
  }
}

void 
neural_datapoint ()
{
  float * outputs;
  float inputs[n_inputs];
  int i;

  inputs[ 0] = avg_speed[1];
  inputs[ 1] = avg_speed[2];
  inputs[ 2] = avg_speed[3];
  inputs[ 3] = avg_speed[4];

  inputs[ 4] = avg_pressure[1];
  inputs[ 5] = avg_pressure[2];
  inputs[ 6] = avg_pressure[3];
  inputs[ 7] = avg_pressure[4];

  inputs[ 8] = avg_x[1] - avg_x[2];
  inputs[ 9] = avg_x[2] - avg_x[3];
  inputs[10] = avg_x[3] - avg_x[4];

  inputs[11] = avg_y[1] - avg_y[2];
  inputs[12] = avg_y[2] - avg_y[3];
  inputs[13] = avg_y[3] - avg_y[4];

  inputs[14] = clear_dist;

  outputs = trainer_run(ann, inputs);
}

void
neural_notice_clear_image (void)
{
  clear_dist = 10.0;
}

void
neural_init (void)
{
  ann = trainer_create_from_file("nntrainer.dat");
  if (ann) 
    {
      g_print ("Loaded ANN from file\n");
    }
  else
    {
      g_print ("Creating new ANN\n");
      ann = trainer_create(n_inputs, n_outputs);
    }
}

void
neural_finish (void)
{
  trainer_save(ann, "nntrainer.dat");
  g_print ("Saved ANN to file\n");
  trainer_destroy(ann);
}
