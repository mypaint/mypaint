// lfd - lowpass filtered derivative
// 2007 Martin Renold, public domain

#include "lfd.h"
#include <math.h>

void lfd_reset(lfd_t * lfd)
{
  lfd->initialized = 0;
  lfd->filtered_signal = 0;
  lfd->filtered_derivative = 0;
}

void lfd_update(lfd_t * lfd, double dtime, double input)
{
  if (lfd->initialized) {
    // running
    double fac;
    if (lfd->time_constant > 0) {
      fac = exp(-dtime/lfd->time_constant);
    } else {
      fac = 0;
    }
    lfd->filtered_signal = fac*lfd->filtered_signal + (1.0-fac)*input;
    lfd->filtered_derivative = input - lfd->filtered_signal;
  } else {
    lfd->filtered_signal = input;
    lfd->filtered_derivative = 0;
    lfd->initialized = 1;
  }
  lfd->input_last = input;
}
