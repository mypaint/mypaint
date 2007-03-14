#include "stroke_recorder.h"

// TODO: encode only diff; lossless compression

GString* event_array_to_string (GArray * ea)
{
  GString * bs;
  bs = g_string_new("1"); // version identifier
  int i;
  for (i=0; i<ea->len; i++) {
    StrokeEvent *e;
    e = &g_array_index (ea, StrokeEvent, i);
    BS_WRITE_INT32 (e->dtime);
    BS_WRITE_FLOAT (e->x);
    BS_WRITE_FLOAT (e->y);
    BS_WRITE_FLOAT (e->pressure);
  }
  return bs;
}

GArray* string_to_event_array (GString * bs)
{
  char * p = bs->str;
  if (bs->len <= 0) {
    g_print ("Empty event string\n");
    return NULL;
  }
  if (*p++ != '1') {
    g_print ("Unknown version ID\n");
    return NULL;
  }
  GArray * ea;
  ea = g_array_new (FALSE, FALSE, sizeof(StrokeEvent));
  while (p<bs->str+bs->len) {
    StrokeEvent e;
    BS_READ_INT32 (e.dtime);
    BS_READ_FLOAT (e.x);
    BS_READ_FLOAT (e.y);
    BS_READ_FLOAT (e.pressure);
    g_array_append_val (ea, e);
  }
  return ea;
}
