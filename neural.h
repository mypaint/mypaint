void neural_load (void);
void neural_save (void);

void neural_process_movement (float dt, float x, float y, float pressure, float dist);
void neural_notice_clear_image (void);

float neural_predict_brushsize ();
void neural_set_current_brushsize (float size);

void neural_init (void);
void neural_finish (void);
