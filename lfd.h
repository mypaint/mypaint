// lfd - lowpass filtered derivative
// 2007 Martin Renold, public domain
//
// Input: event based signal - (timestamp, value) tuples
// Output: lowpass filtered signal and its derivative
//
// Remember to check the bordercases: long eventless periods and
// initialization.

typedef struct {
  // parameters
  double time_constant; // = 1.0/cutoff_frequency
  // internal states
  double input_last;
  int initialized;
  // outputs
  double filtered_signal;
  double filtered_derivative;
} lfd_t;

void lfd_reset(lfd_t * lfd);
void lfd_update(lfd_t * lfd, double dtime, double input);
