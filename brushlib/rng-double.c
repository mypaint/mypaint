/*    This program by D E Knuth is in the public domain and freely copyable.
 *    It is explained in Seminumerical Algorithms, 3rd edition, Section 3.6
 *    (or in the errata to the 2nd edition --- see
 *        http://www-cs-faculty.stanford.edu/~knuth/taocp.html
 *    in the changes to Volume 2 on pages 171 and following).              */

/*    N.B. The MODIFICATIONS introduced in the 9th printing (2002) are
      included here; there's no backwards compatibility with the original. */

/*    This version also adopts Brendan McKay's suggestion to
      accommodate naive users who forget to call ranf_start(seed).         */

/*    If you find any bugs, please report them immediately to
 *                 taocp@cs.stanford.edu
 *    (and you will be rewarded if the bug is genuine). Thanks!            */

/************ see the book for explanations and caveats! *******************/
/************ in particular, you need two's complement arithmetic **********/

/* This version has been changed by Jon Nordby to allow to create multiple
 * independent generator objects. All changes made to this file are considered
 * to be in the public domain. */

#include "rng-double.h"

#include <stdlib.h>

/* the following routines are adapted from exercise 3.6--15 */
/* after calling ranf_start, get new randoms by, e.g., "x=ranf_arr_next()" */

#if 0
/* original settings */
#define QUALITY 1009 /* recommended quality level for high-res use */
#define TT  70   /* guaranteed separation between streams */
#define KK 100                     /* the long lag */
#define LL  37                     /* the short lag */
#else
/* low quality settings, seems to work for MyPaint */
/* (Disclaimer: I don't understand what those numbers do, I just reduced them. --maxy) */
#define QUALITY 19
#define TT  7
#define KK 10
#define LL  7
#endif

#define is_odd(s) ((s)&1)
#define mod_sum(x,y) (((x)+(y))-(int)((x)+(y)))   /* (x+y) mod 1.0 */

const double ranf_arr_dummy=-1.0;
const double ranf_arr_started=-1.0;

struct _RngDouble {
    double ran_u[KK];           /* the generator state */
    double ranf_arr_buf[QUALITY];
    double *ranf_arr_ptr; /* the next random fraction, or -1 */
};

void
rng_double_get_array(RngDouble *self, double aa[], int n)
{
  register int i,j;
  for (j=0;j<KK;j++) aa[j]=self->ran_u[j];
  for (;j<n;j++) aa[j]=mod_sum(aa[j-KK],aa[j-LL]);
  for (i=0;i<LL;i++,j++) self->ran_u[i]=mod_sum(aa[j-KK],aa[j-LL]);
  for (;i<KK;i++,j++) self->ran_u[i]=mod_sum(aa[j-KK],self->ran_u[i-LL]);
}


RngDouble *
rng_double_new(long seed)
{
  RngDouble *self = (RngDouble *)malloc(sizeof(RngDouble));

  self->ranf_arr_ptr=(double *)&ranf_arr_dummy;

  rng_double_set_seed(self, seed);

  return self;
}

void
rng_double_free(RngDouble *self)
{
    free(self);
}

void
rng_double_set_seed(RngDouble *self, long seed)
{
  register int t,s,j;
  double u[KK+KK-1];
  double ulp=(1.0/(1L<<30))/(1L<<22);               /* 2 to the -52 */
  double ss=2.0*ulp*((seed&0x3fffffff)+2);

  for (j=0;j<KK;j++) {
    u[j]=ss;                                /* bootstrap the buffer */
    ss+=ss; if (ss>=1.0) ss-=1.0-2*ulp;  /* cyclic shift of 51 bits */
  }
  u[1]+=ulp;                     /* make u[1] (and only u[1]) "odd" */
  for (s=seed&0x3fffffff,t=TT-1; t; ) {
    for (j=KK-1;j>0;j--)
      u[j+j]=u[j],u[j+j-1]=0.0;                         /* "square" */
    for (j=KK+KK-2;j>=KK;j--) {
      u[j-(KK-LL)]=mod_sum(u[j-(KK-LL)],u[j]);
      u[j-KK]=mod_sum(u[j-KK],u[j]);
    }
    if (is_odd(s)) {                             /* "multiply by z" */
      for (j=KK;j>0;j--) u[j]=u[j-1];
      u[0]=u[KK];                    /* shift the buffer cyclically */
      u[LL]=mod_sum(u[LL],u[KK]);
    }
    if (s) s>>=1; else t--;
  }
  for (j=0;j<LL;j++) self->ran_u[j+KK-LL]=u[j];
  for (;j<KK;j++) self->ran_u[j-LL]=u[j];
  for (j=0;j<10;j++) rng_double_get_array(self, u,KK+KK-1);  /* warm things up */
  self->ranf_arr_ptr=(double *)&ranf_arr_started;
}


double
rng_double_cycle(RngDouble *self)
{
  rng_double_get_array(self, self->ranf_arr_buf, QUALITY);
  self->ranf_arr_buf[KK]=-1;
  self->ranf_arr_ptr=self->ranf_arr_buf+1;
  return self->ranf_arr_buf[0];
}

double
rng_double_next(RngDouble *self)
{
  return ((*self->ranf_arr_ptr>=0) ? *(self->ranf_arr_ptr)++ : rng_double_cycle(self));
}
