#include "helpers.h"
#include <glib.h>

// stolen from the gimp (noisify.c)

/*
 * Return a Gaussian (aka normal) random variable.
 *
 * Adapted from ppmforge.c, which is part of PBMPLUS.
 * The algorithm comes from:
 * 'The Science Of Fractal Images'. Peitgen, H.-O., and Saupe, D. eds.
 * Springer Verlag, New York, 1988.
 *
 * It would probably be better to use another algorithm, such as that
 * in Knuth
 */
gdouble gauss_noise (void)
{
  gint i;
  gdouble sum = 0.0;

  for (i = 0; i < 4; i++)
    sum += g_random_int_range (0, 0x7FFF);

  return sum * 5.28596089837e-5 - 3.46410161514;
}
