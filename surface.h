/* A drawing surface (planned: with limited undo functionality)
   Copyright 2004 Martin Renold <martinxyz@gmx.ch>
   Released as public domain.
*/

#ifndef __surface_h__
#define __surface_h__

#include <gtk/gtk.h>

#define sqr(x) ((x)*(x))

#define SIZE 512

#define BLOCKBITS 7
#define BLOCKSIZE (1 << BLOCKBITS)
#define BLOCKWRAP (BLOCKSIZE-1)
/* To keep physically close memory together, pixels are numbered in
   blocks of BLOCKSIZE*BLOCKSIZE, like this (for BLOCKBITS = 2):
   +-------------+-------------+--
   |  0  1  2  3 | 16 17 18 19 | ...
   |  4  5  6  7 | 20 21 22 23 | 
   |  8  9 10 11 | 24 25 26 27 |
   | 12 13 14 15 | 28 29 30 31 |
   +-------------+-------------+--
   | ...         |             |
   This helps to keep them in the CPU cache.

   Optimal BLOCKBITS value depends on CPU cache (and how often actual
   random access occurs). 
   Expected speedup by this is 11% (measured in another program)
 */
#define PixelXY(surf, x, y) (surf->rgb + 3*( /* Dear compiler. Please simplify this all. */ \
/* block position */   ((x)&(~BLOCKWRAP))*BLOCKSIZE + ((y)&(~BLOCKWRAP))*SIZE \
/* within a block */ + BLOCKSIZE*((y)&BLOCKWRAP) + ((x)&BLOCKWRAP)))

typedef struct {
  int w, h; /* fixed to SIZE*SIZE for now */
  guchar * rgb; /* data, memory not linear (see above) */
  GtkWidget * widget; /* where to queue draws when changed */
} Surface;

Surface * new_surface (int w, int h);
void free_surface (Surface * s);
void surface_clear (Surface * s);
void surface_renderpattern (Surface * s);
void surface_render (Surface * s, guchar * dst, int rowstride, int x0, int y0, int w, int h);

#endif
