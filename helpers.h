#include <glib.h>
// gaussian noise, mean 0 variance 1
extern gdouble gauss_noise (void);

#define ROUND(x) ((int) ((x) + 0.5))

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

