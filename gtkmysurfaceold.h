/* This file is part of MyPaint.
 * Copyright (C) 2007 by Martin Renold <martinxyz@gmx.ch>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License.
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY. See the COPYING file for more details.
 */

#ifndef __GTK_MY_SURFACE_OLD_H__
#define __GTK_MY_SURFACE_OLD_H__

#include <glib.h>
#include <glib-object.h>
#include <gdk/gdk.h>
#include "gtkmysurface.h"
#include "helpers.h"

#define GTK_TYPE_MY_SURFACE_OLD            (gtk_my_surface_old_get_type ())
#define GTK_MY_SURFACE_OLD(obj)            (G_TYPE_CHECK_INSTANCE_CAST ((obj), GTK_TYPE_MY_SURFACE_OLD, GtkMySurfaceOld))
#define GTK_MY_SURFACE_OLD_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST ((klass), GTK_TYPE_MY_SURFACE_OLD, GtkMySurfaceOldClass))
#define GTK_IS_MY_SURFACE_OLD(obj)         (G_TYPE_CHECK_INSTANCE_TYPE ((obj), GTK_TYPE_MY_SURFACE_OLD))
#define GTK_IS_MY_SURFACE_OLD_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE ((klass), GTK_TYPE_MY_SURFACE_OLD))
#define GTK_MY_SURFACE_OLD_GET_CLASS(obj)  (G_TYPE_INSTANCE_GET_CLASS ((obj), GTK_TYPE_MY_SURFACE_OLD, GtkMySurfaceOldClass))


typedef struct _GtkMySurfaceOld       GtkMySurfaceOld;
typedef struct _GtkMySurfaceOldClass  GtkMySurfaceOldClass;

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

struct _GtkMySurfaceOld
{
  GtkMySurface parent;

  guchar * rgb; /* data, memory not linear (see above) */
  unsigned char xsize_shl;
  int w, h;
  int block_w, block_h;
};

struct _GtkMySurfaceOldClass
{
  GtkMySurfaceClass parent_class;

  // any emitted signals? notification of change maxrects?
  //void (*dragging_finished) (GtkMySurfaceOld *mdw);
};

GType      gtk_my_surface_old_get_type   (void) G_GNUC_CONST;

GtkMySurfaceOld* gtk_my_surface_old_new        (int w, int h);

void gtk_my_surface_old_renderpattern (GtkMySurfaceOld * s);
void gtk_my_surface_old_render (GtkMySurfaceOld * s, guchar * dst, int rowstride, int x0, int y0, int w, int h, int bpp);
void gtk_my_surface_old_render_zoom (GtkMySurfaceOld * s, guchar * dst, int rowstride, float x0, float y0, int w, int h, int bpp, float one_over_zoom);
void gtk_my_surface_old_load (GtkMySurfaceOld * s, guchar * src, int rowstride, int w, int h, int bpp);
void gtk_my_surface_old_get_nonwhite_region (GtkMySurfaceOld * s, Rect * r);

#endif /* __GTK_MY_SURFACE_OLD_H__ */
