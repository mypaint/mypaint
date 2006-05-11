/* A drawing surface (planned: with limited undo functionality)

   Copyright 2004 Martin Renold <martinxyz@gmx.ch>
   I place this file into the public domain. Do with it whatever you want.
*/

#ifndef __surface_h__
#define __surface_h__

#include "helpers.h"

#define sqr(x) ((x)*(x))

// FIXME: big waste of memory... however since it's arranged so near
// things stay together, it should not matter much - even if the OS
// would swap some parts out. Bad idea, anyway.

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
#define PixelXY(surf, x, y) ((surf)->rgb + 3*( /* Dear compiler. Please simplify this all. */ \
/* block position */   ((x)&(~BLOCKWRAP))*BLOCKSIZE + (((y)&(~BLOCKWRAP))<<(surf)->xsize_shl) \
/* within a block */ + BLOCKSIZE*((y)&BLOCKWRAP) + ((x)&BLOCKWRAP)))

typedef struct {
  guchar * rgb; /* data, memory not linear (see above) */
  unsigned char xsize_shl;
  int w, h;
  int block_w, block_h;
} Surface;

Surface * new_surface (int w, int h);
void free_surface (Surface * s);
void surface_clear (Surface * s);
void surface_renderpattern (Surface * s);
void surface_render (Surface * s, guchar * dst, int rowstride, int x0, int y0, int w, int h, int bpp);
void surface_render_zoom (Surface * s, guchar * dst, int rowstride, float x0, float y0, int w, int h, int bpp, float one_over_zoom);
void surface_load (Surface * s, guchar * src, int rowstride, int w, int h, int bpp);
void surface_get_nonwhite_region (Surface * s, Rect * r);

#endif
