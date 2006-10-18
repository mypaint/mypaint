#include <stdio.h>
#include <string.h>
#include <glib.h>
#include <math.h>
#include "gtkmysurfaceold.h"


static void gtk_my_surface_old_class_init    (GtkMySurfaceOldClass *klass);
static void gtk_my_surface_old_init          (GtkMySurfaceOld      *b);
static void gtk_my_surface_old_finalize (GObject *object);
static void gtk_my_surface_old_clear (GtkMySurface *s);

static gpointer parent_class;

GType
gtk_my_surface_old_get_type (void)
{
  static GType type = 0;

  if (!type)
    {
      static const GTypeInfo info =
      {
	sizeof (GtkMySurfaceOldClass),
	NULL,		/* base_init */
	NULL,		/* base_finalize */
	(GClassInitFunc) gtk_my_surface_old_class_init,
	NULL,		/* class_finalize */
	NULL,		/* class_data */
	sizeof (GtkMySurfaceOld),
	0,		/* n_preallocs */
	(GInstanceInitFunc) gtk_my_surface_old_init,
      };

      type =
	g_type_register_static (GTK_TYPE_MY_SURFACE, "GtkMySurfaceOld",
				&info, 0);
    }

  return type;
}

static void
gtk_my_surface_old_class_init (GtkMySurfaceOldClass *class)
{
  GObjectClass *gobject_class = G_OBJECT_CLASS (class);
  parent_class = g_type_class_peek_parent (class);
  gobject_class->finalize = gtk_my_surface_old_finalize;
  GTK_MY_SURFACE_CLASS (class)->clear = gtk_my_surface_old_clear;
}

static void
gtk_my_surface_old_init (GtkMySurfaceOld *s)
{
  s->w = 0;
  s->h = 0;
  s->rgb = NULL;
}

static void
gtk_my_surface_old_finalize (GObject *object)
{
  GtkMySurfaceOld * s = GTK_MY_SURFACE_OLD (object);
  g_free (s->rgb);
  s->rgb = NULL;
  G_OBJECT_CLASS (parent_class)->finalize (object);
}

GtkMySurfaceOld*
gtk_my_surface_old_new  (int w, int h)
{
  GtkMySurfaceOld * s = g_object_new (GTK_TYPE_MY_SURFACE_OLD, NULL);
  //g_print ("gtk_my_surface_old_new (This should get called.)\n");

  s->w = w;
  s->h = h;

  // ugh. hope that's all correct now
  s->block_h = (h-1) / BLOCKSIZE + 1;
  
  s->xsize_shl = BLOCKBITS;
  while ((1 << s->xsize_shl) < w) s->xsize_shl++;

  s->block_w = 1 << (s->xsize_shl - BLOCKBITS);

  /*
  g_print ("requested: %d %d\n", s->w, s->h);
  g_print ("blocksize: %d %d\n", s->block_w, s->block_h);
  g_print ("finally  : %d %d\n", s->block_w*BLOCKSIZE, s->block_h*BLOCKSIZE);
  */

  g_assert (s->block_w * BLOCKSIZE >= w);
  g_assert (s->block_h * BLOCKSIZE >= h);

  s->rgb = g_new0 (guchar, 3*s->block_w*BLOCKSIZE*s->block_h*BLOCKSIZE);
  return s;
}

void
gtk_my_surface_old_renderpattern (GtkMySurfaceOld * s)
{
  int x, y, r, g, b;
  guchar *rgb;

  for (y = 0; y < s->h; y++) {
    for (x = 0; x < s->w; x++) {
      rgb = PixelXY(s, x, y);
      r = x % 256;
      g = y % 256;
      b = (x*x+y*y) % 256;
      rgb[3*x + 0] = r;
      rgb[3*x + 1] = g;
      rgb[3*x + 2] = b;
    }
  }
}

static void
gtk_my_surface_old_clear (GtkMySurface *s_)
{
  GtkMySurfaceOld *s = GTK_MY_SURFACE_OLD(s_);
  // clear rgb buffer to white
  memset (s->rgb, 255, s->block_w*BLOCKSIZE*s->block_h*BLOCKSIZE*3);
}

void
gtk_my_surface_old_render (GtkMySurfaceOld * s, guchar * dst, int rowstride, int x0, int y0, int w, int h, int bpp)
{
  // WARNING: Code duplication with surface_render_zoom below
  // could be optimized much, important if big brush is used
  int x, y, add;
  if (bpp == 3*8) {
    add = 3;
  } else if (bpp == 4*8) {
    add = 4;
  } else {
    g_assert (0);
    return;
  }

  guchar white[3];
  white[0] = 255;
  white[1] = 255;
  white[2] = 255;

  guchar * rgb_line = dst;
  guchar * rgb_dst;
  guchar * rgb_src;
  for (y = y0; y < y0 + h; y++) {
    rgb_dst = rgb_line;
    for (x = x0; x < x0 + w; x++) {
      if (x < 0 || y < 0 || x >= s->w || y >= s->h) {
        rgb_src = white;
      } else {
        rgb_src = PixelXY(s, x, y);
      }
      rgb_dst[0] = rgb_src[0];
      rgb_dst[1] = rgb_src[1];
      rgb_dst[2] = rgb_src[2];
      rgb_dst += add;
    }
    rgb_line += rowstride;
  }
}

void
gtk_my_surface_old_render_zoom (GtkMySurfaceOld * s, guchar * dst, int rowstride, float x0, float y0, int w, int h, int bpp, float one_over_zoom)
{
  // WARNING: Code duplication with surface_render above
  // could be optimized much, important if big brush is used
  int x, y, add;
  int x_final, y_final;
  if (bpp == 3*8) {
    add = 3;
  } else if (bpp == 4*8) {
    add = 4;
  } else {
    g_assert (0);
    return;
  }

  guchar white[3];
  white[0] = 255;
  white[1] = 255;
  white[2] = 255;

  guchar * rgb_line = dst;
  guchar * rgb_dst;
  guchar * rgb_src;
  for (y = 0; y < h; y++) {
    rgb_dst = rgb_line;
    for (x = 0; x < w; x++) {
      x_final = (x0+x) * one_over_zoom + 0.5;
      y_final = (y0+y) * one_over_zoom + 0.5;
      if (x_final < 0 || y_final < 0 || x_final >= s->w || y_final >= s->h) {
        rgb_src = white;
      } else {
        rgb_src = PixelXY(s, x_final, y_final);
      }
      rgb_dst[0] = rgb_src[0];
      rgb_dst[1] = rgb_src[1];
      rgb_dst[2] = rgb_src[2];
      rgb_dst += add;
    }
    rgb_line += rowstride;
  }
}

void
gtk_my_surface_old_load (GtkMySurfaceOld * s, guchar * src, int rowstride, int w, int h, int bpp)
{
  int x, y, add;
  if (bpp == 3*8) {
    add = 3;
  } else if (bpp == 4*8) {
    add = 4;
  } else {
    g_assert (0);
    return;
  }

  if (w > s->w) w = s->w;
  if (h > s->h) h = s->h;

  guchar * rgb_line = src;
  guchar * rgb_dst;
  guchar * rgb_src;
  for (y = 0; y < 0 + h; y++) {
    rgb_src = rgb_line;
    for (x = 0; x < 0 + w; x++) {
      rgb_dst = PixelXY(s, x, y);
      rgb_dst[0] = rgb_src[0];
      rgb_dst[1] = rgb_src[1];
      rgb_dst[2] = rgb_src[2];
      rgb_src += add;
    }
    rgb_line += rowstride;
  }
}

void
gtk_my_surface_old_get_nonwhite_region (GtkMySurfaceOld * s, Rect * r)
{
  int x, y;
  r->w = 0;

  for (y = 0; y < s->h; y++) {
    for (x = 0; x < s->w; x++) {
      guchar * rgb;
      rgb = PixelXY(s, x, y);
      if (rgb[0] != 255 || rgb[1] != 255 || rgb[2] != 255) {
        ExpandRectToIncludePoint(r, x, y);
      }
    }
  }

  if (r->w == 0) {
    // all empty; make it easy for the other code
    r->x = 0;
    r->y = 0;
    r->w = 1;
    r->h = 1;
  }
}
