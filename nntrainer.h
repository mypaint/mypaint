/* Easy interface for online-learning with FANN.
 * (c) 2004 Martin Renold, public domain
 *
 * Automatically scale/center data, adapt learning rate, choose number
 * of hidden units, restart training with new weights, remove old data
 */
#ifndef __nntrainer_h__
#define __nntrainer_h__
#include <fann.h>

struct trainer {
  int ni, no; /* number of inputs/outputs */
  int data_used, data_size;
  fann_type * data;
  fann_type * mean;
  fann_type * std;
  struct fann * ann;
};

struct trainer * trainer_create(int num_inputs, int num_outputs);
struct trainer * trainer_create_from_file(const char *filename);
void trainer_save(struct trainer * t, const char *filename);
void trainer_save_textdata(struct trainer * t, const char *filename);
void trainer_destroy(struct trainer * t);

/* once max_data is reached, this replaces a random old data piece */
void trainer_add_data(struct trainer * t, fann_type * inputs, fann_type * outputs);
void trainer_train(struct trainer * t);

fann_type * trainer_run(struct trainer * t, fann_type * inputs);

#endif 
