#ifndef __helpers_h__
#define __helpers_h__

#include <glib.h>
#include <assert.h>
#include "random.h"

gdouble synced_random_double (void);
gdouble synced_gauss_noise (void);

#define ROUND(x) ((int) ((x) + 0.5))
#define SIGN(x) ((x)>0?1:(-1))
#define SQR(x) ((x)*(x))

#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif
#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

void    gimp_rgb_to_hsv_int	(gint    *red         /* returns hue        */,
				 gint    *green       /* returns saturation */,
				 gint    *blue        /* returns value      */);
void    gimp_hsv_to_rgb_int	(gint    *hue         /* returns red        */,
				 gint    *saturation  /* returns green      */,
				 gint    *value       /* returns blue       */);

typedef struct { int x, y, w, h; } Rect;
void ExpandRectToIncludePoint(Rect * r, int x, int y);

// Binary String (GString) read/write. Those macros are full of
// assumptions. They convert to/from big endian.
typedef union {           gint16 i; gchar c[2]; } uni16;
typedef union { float f;  gint32 i; gchar c[4]; } uni32;
typedef union { double d; gint64 i; gchar c[8]; } uni64;
#define BS_WRITE_INT16(value)  { uni16 u; u.i = value; u.i = GINT16_TO_BE(u.i); g_string_append_len (bs, u.c, 2); }
#define BS_WRITE_INT32(value)  { uni32 u; u.i = value; u.i = GINT32_TO_BE(u.i); g_string_append_len (bs, u.c, 4); }
#define BS_WRITE_INT64(value)  { uni64 u; u.i = value; u.i = GINT64_TO_BE(u.i); g_string_append_len (bs, u.c, 8); }
#define BS_WRITE_FLOAT(value)  { uni32 u; u.f = value; u.i = GINT32_TO_BE(u.i); g_string_append_len (bs, u.c, 4); }
#define BS_WRITE_DOUBLE(value) { uni64 u; u.f = value; u.i = GINT64_TO_BE(u.i); g_string_append_len (bs, u.c, 8); }
#define BS_WRITE_CHAR(value)   { g_string_append_c (bs, value); }

#define BS_READ_INT16(result)  { result = GINT16_FROM_BE(*((gint16*)p)); p+=2; }
#define BS_READ_INT32(result)  { result = GINT32_FROM_BE(*((gint32*)p)); p+=4; }
#define BS_READ_INT64(result)  { result = GINT64_FROM_BE(*((gint64*)p)); p+=8; }
#define BS_READ_FLOAT(result)  { uni32 tmp; BS_READ_INT32(tmp.i); result = tmp.f; }
#define BS_READ_DOUBLE(result) { uni64 tmp; BS_READ_INT64(tmp.i); result = tmp.d; }
#define BS_READ_CHAR(result)   { result = *p++; }

#endif
