/* A drawing surface with limited undo functionality.
   Copyright 2004 Martin Renold <martinxyz@gmx.ch>
   Released under GPL.
*/

#define sqr(x) ((x)*(x))

typedef unsigned char byte;
typedef unsigned short u16;
typedef unsigned int u32;

typedef struct {
  int w, h;
  int rowstride;
  byte * rgb;
} Surface;

typedef struct {
  double radius;
  byte color[3];
  double opaque;
} Brush;

Surface * new_surface (int w, int h);
void free_surface (Surface * s);
void surface_clear (Surface * s);
void surface_draw (Surface * s, double x, double y, Brush * b);
void surface_renderpattern (Surface * s);
