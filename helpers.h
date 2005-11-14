#ifndef __helpers_h__
#define __helpers_h__

#include <glib.h>
#include <assert.h>
// gaussian noise, mean 0 variance 1
extern gdouble gauss_noise (void);

#define ROUND(x) ((int) ((x) + 0.5))
#define SIGN(x) ((x)>0?1:(-1))

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


#endif
