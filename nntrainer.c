/* (c) 2004 Martin Renold, public domain
 */

#include "nntrainer.h"
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define IOs(t) ((t)->ni+(t)->no)

struct trainer * trainer_create(int num_inputs, int num_outputs)
{
  struct trainer * t;
  t = calloc(1, sizeof(struct trainer));
  t->ni = num_inputs;
  t->no = num_outputs;
  assert(t->ni > 0 && t->no > 0);
  t->data_size = 2000;
  return t;
}

void trainer_save(struct trainer * t, const char *filename)
{
  FILE * f;
  int check;
  f = fopen(filename, "w");
  /* FIXME: endian problems */
  fwrite(t, sizeof(struct trainer), 1, f);
  if (t->data) fwrite(t->data, IOs(t)*t->data_size*sizeof(fann_type), 1, f);
  if (t->mean) fwrite(t->mean, IOs(t)*sizeof(fann_type), 1, f);
  if (t->std) fwrite(t->std, IOs(t)*sizeof(fann_type), 1, f);
  if (t->ann) {
    char * newname;
    newname = malloc(strlen(filename)+4+1);
    strcpy(newname, filename);
    strcat(newname, ".nn0");
    fann_save(t->ann, newname);
    free(newname);
  }
  check = 123456;
  fwrite(&check, sizeof(int), 1, f);
  fclose(f);
}

struct trainer * trainer_create_from_file(const char *filename)
{
  FILE * f;
  int check;
  struct trainer * t;
  t = trainer_create(1, 1);
  f = fopen(filename, "r");
  if (!f) return NULL;
  fread(t, sizeof(struct trainer), 1, f);
  if (t->data) {
    t->data = malloc(IOs(t)*t->data_size*sizeof(fann_type));
    fread(t->data, IOs(t)*t->data_size*sizeof(fann_type), 1, f);
  }
  if (t->mean) {
    t->mean = malloc(IOs(t)*sizeof(fann_type));
    fread(t->mean, IOs(t)*sizeof(fann_type), 1, f);
  }
  if (t->std) {
    t->std = malloc(IOs(t)*sizeof(fann_type));
    fread(t->std, IOs(t)*sizeof(fann_type), 1, f);
  }
  if (t->ann) {
    char * newname;
    newname = malloc(strlen(filename)+4+1);
    strcpy(newname, filename);
    strcat(newname, ".nn0");
    t->ann = fann_create_from_file(newname);
    assert(t->ann);
    free(newname);
  }
  fread(&check, sizeof(int), 1, f);
  assert(check == 123456);
  fclose(f);
  return t;
}

void trainer_destroy(struct trainer * t)
{
  free(t->data);
  free(t->mean);
  free(t->std);
  if (t->ann) fann_destroy(t->ann);
}

void trainer_add_data(struct trainer * t, fann_type * inputs, fann_type * outputs)
{
  if (!t->data) {
    t->data = malloc(IOs(t)*t->data_size*sizeof(fann_type));
    assert(t->data_used == 0);
    t->data_used++;
  } else {
    /* move old t->data[0] away */
    int dst;
    assert(t->data_used > 0);
    if (t->data_used < t->data_size) {
      dst = t->data_used++;
    } else {
      assert(t->data_used == t->data_size);
      /* FIXME: who does srandom(time)? */
      dst = rand() % t->data_size;
    }
    memcpy(t->data + dst*IOs(t), t->data, sizeof(fann_type)*IOs(t));
  }
  memcpy(t->data, inputs, sizeof(fann_type)*t->ni);
  memcpy(t->data + t->ni, outputs, sizeof(fann_type)*t->no);
}

void trainer_reset_training(struct trainer * t)
{
  int i, j, N;
  N = t->data_used / 2; /* training set */

  if (!t->mean) t->mean = malloc(IOs(t)*sizeof(fann_type));
  if (!t->std) t->std = malloc(IOs(t)*sizeof(fann_type));

  for (j=0; j<IOs(t); j++) {
    t->mean[j] = 0;
    t->std[j] = 0;
  }
  for (i=0; i<N; i++) {
    fann_type * data = t->data + i*IOs(t);
    for (j=0; j<IOs(t); j++, data++) {
      t->mean[j] += (*data)/N;
    }
  }
  for (i=0; i<N; i++) {
    fann_type * data = t->data + i*IOs(t);
    for (j=0; j<IOs(t); j++, data++) {
      fann_type deriv;
      deriv = *data - t->mean[j];
      deriv *= deriv;
      if (j < t->ni) {
        /* inputs: can be any range, but use standard derivation 1 */
        t->std[j] += deriv/N;
      } else {
        /* outputs: must be strictly between 0 and 1 */
        t->std[j] = MAX(deriv, t->std[j]);
      }
    }
  }
  for (j=0; j<IOs(t); j++) {
    t->std[j] = sqrt(t->std[j]);
    if (t->std[j] < 0.00001) t->std[j] = 0.00001;
  }

  if (t->ann) fann_destroy(t->ann);
  /* FIXME: test optimal number of hidden nodes and number of layers etc. */
  t->ann = fann_create(1.0, 0.7, 3, t->ni, (t->ni+t->no+1)/2, t->no);
}


fann_type scale_input(fann_type input, fann_type mean, fann_type std)
{
  return (input - mean) / std;
}

fann_type scale_output(fann_type output, fann_type mean, fann_type std)
{
  return ((output - mean) / std) * 0.4 + 0.5; /* final range: 0.1 ... 0.9 */
}

fann_type unscale_output(fann_type output, fann_type mean, fann_type std)
{
  return ((output - 0.5) / 0.4) * std + mean;
}

void trainer_train_step(struct trainer * t)
{
  int N, i, j;
  fann_type * io;
  io = malloc(IOs(t)*sizeof(fann_type));
  N = t->data_used/2;
  for (i=0; i<N; i++) {
    memcpy(io, t->data + i*IOs(t), IOs(t)*sizeof(fann_type));
    for (j=0; j<IOs(t); j++) {
      if (j<t->ni) {
        io[j] = scale_input(io[j], t->mean[j], t->std[j]);
      } else {
        io[j] = scale_output(io[j], t->mean[j], t->std[j]);
      }
    }
    fann_train(t->ann, io, io + t->ni);
  }
  free(io);
}

fann_type trainer_test_step(struct trainer * t)
{
  int N, i, j;
  fann_type * io;
  io = malloc(IOs(t)*sizeof(fann_type));
  N = t->data_used/2;
  fann_reset_MSE(t->ann);
  for (i=N; i<t->data_used; i++) {
    memcpy(io, t->data + i*IOs(t), IOs(t)*sizeof(fann_type));
    for (j=0; j<IOs(t); j++) {
      if (j<t->ni) {
        io[j] = scale_input(io[j], t->mean[j], t->std[j]);
      } else {
        io[j] = scale_output(io[j], t->mean[j], t->std[j]);
      }
    }
    fann_test(t->ann, io, io + t->ni);
  }
  free(io);
  return fann_get_MSE(t->ann);
}

void trainer_train(struct trainer * t)
{
  int i;
  float eta;
  fann_type mse, old_mse;
  assert(t->data);
  assert(t->data_used > 8);
  trainer_reset_training(t);
  i = 0;
  mse = -1;
  /* train until overfitting */
  /*eta = 3.0;*/
  eta = 0.001;
  do {
    fann_set_learning_rate(t->ann, eta); 
    old_mse = mse;
    mse = trainer_test_step(t);
    printf("%d %f, %f\n", i, mse, eta);
    trainer_train_step(t);
    i++;
    if (old_mse == -1 || mse < old_mse*0.99 /* more than 1% better is considered progress */) {
      eta *= 1.02;
    } else {
      eta *= 0.5;
    }
  } while (eta > 0.00001);
}

fann_type * trainer_run(struct trainer * t, fann_type * original_inputs)
{
  int j;
  fann_type * inputs;
  fann_type * outputs;
  assert(t->data);
  assert(t->ann);

  inputs = malloc(t->ni*sizeof(fann_type));
  memcpy(inputs, original_inputs, t->ni*sizeof(fann_type));

  for (j=0; j<t->ni; j++) {
    inputs[j] = scale_input(inputs[j], t->mean[j], t->std[j]);
  }
  outputs = fann_run(t->ann, inputs);
  for (j=0; j<t->no; j++) {
    outputs[j] = unscale_output(outputs[j], t->mean[j+t->ni], t->std[j+t->ni]);
  }
  free(inputs);
  return outputs;
}

fann_type test_func(fann_type x, fann_type y)
{
  return sin(x+y) * x + y;
}

int test_nntrainer()
{
  int i;
  struct trainer * t;
  fann_type inputs[2];
  fann_type outputs[1];
  t = trainer_create(2, 1);
  for (i=0; i<5000; i++) {
    inputs[0] = (rand() % 500)*0.002;
    inputs[1] = (rand() % 500)*0.0001 - 1000;
    outputs[0] = test_func(inputs[0], inputs[1]);
    if (rand() % 3 == 0) outputs[0] += ((rand() % 100)-50) / 5000.0; /* noise */
    trainer_add_data(t, inputs, outputs);
  }
  trainer_train(t);
  trainer_save(t, "delme.trainer");
  trainer_destroy(t);
  t = trainer_create_from_file("delme.trainer");
  for (i=0; i<10; i++) {
    fann_type * guessed_outputs;
    inputs[0] = (rand() % 500)*0.002;
    inputs[1] = (rand() % 500)*0.0001 - 1000;
    guessed_outputs = trainer_run(t, inputs);
    outputs[0] = test_func(inputs[0], inputs[1]);
    printf("%f %f ==> %f (correct: %f)\n", inputs[0], inputs[1], guessed_outputs[0], outputs[0]);
  }
  trainer_destroy(t);
  return 0;
}
