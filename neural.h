void neural_load (void);
void neural_save (void);

void neural_process_movement (float dt, float x, float y, float z, float d_dist, int record_data, int guess_size);
void neural_notice_clear_image (void);

float neural_get_suggested_brushsize ();
void neural_set_current_brushsize (float size);

void neural_init (void);
void neural_finish (void);

void neural_train (void);
