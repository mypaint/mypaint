#include "helpers.h"

typedef struct {
  int dtime;
  float x, y;
  float pressure;
} StrokeEvent;

GString* event_array_to_string (GArray * ea);
GArray* string_to_event_array (GString * bs);
