/*    This program by D E Knuth is in the public domain and freely copyable.
 *    It is explained in Seminumerical Algorithms, 3rd edition, Section 3.6
 *    (or in the errata to the 2nd edition --- see
 *        http://www-cs-faculty.stanford.edu/~knuth/taocp.html
 *    in the changes to Volume 2 on pages 171 and following).              */

/*    N.B. The MODIFICATIONS introduced in the 9th printing (2002) are
      included here; there's no backwards compatibility with the original. */

/*    This version also adopts Brendan McKay's suggestion to
      accommodate naive users who forget to call ran_start(seed).          */

/*    If you find any bugs, please report them immediately to
 *                 taocp@cs.stanford.edu
 *    (and you will be rewarded if the bug is genuine). Thanks!            */

/************ see the book for explanations and caveats! *******************/
/************ in particular, you need two's complement arithmetic **********/

/* This version has been changed by Jon Nordby to allow to create multiple
 * independent generator objects. All changes made to this file are considered
 * to be in the public domain. */

#include "rng-int.h"

#include <malloc.h>

#define KK 100                     /* the long lag */
#define LL  37                     /* the short lag */
#define MM (1L<<30)                 /* the modulus */
#define mod_diff(x,y) (((x)-(y))&(MM-1)) /* subtraction mod MM */

#define QUALITY 1009 /* recommended quality level for high-res use */
#define TT  70   /* guaranteed separation between streams */
#define is_odd(x)  ((x)&1)          /* units bit of x */

const long ran_arr_dummy=-1;
const long ran_arr_started=-1;

struct _RngInt {
    long ran_x[KK];                    /* the generator state */
    long ran_arr_buf[QUALITY];
    long *ran_arr_ptr; /* the next random number, or -1 */
};


/* put n new random numbers in aa */

void
rng_int_get_array(RngInt *self, long aa[],int n)
{
  register int i,j;
  for (j=0;j<KK;j++) aa[j]=self->ran_x[j];
  for (;j<n;j++) aa[j]=mod_diff(aa[j-KK],aa[j-LL]);
  for (i=0;i<LL;i++,j++) self->ran_x[i]=mod_diff(aa[j-KK],aa[j-LL]);
  for (;i<KK;i++,j++) self->ran_x[i]=mod_diff(aa[j-KK],self->ran_x[i-LL]);
}

/* the following routines are from exercise 3.6--15 */
/* after calling ran_start, get new randoms by, e.g., "x=ran_arr_next()" */



/* do this before using ran_array */
RngInt *
rng_int_new(long seed)
{
  RngInt *self = (RngInt *)malloc(sizeof(RngInt));

  self->ran_arr_ptr=(long *)&ran_arr_dummy;

  rng_int_set_seed(self, seed);

  return self;
}

void
rng_int_free(RngInt *self)
{
    free(self);
}

void
rng_int_set_seed(RngInt *self, long seed)
{
  register int t,j;
  long x[KK+KK-1];              /* the preparation buffer */
  register long ss=(seed+2)&(MM-2);
  for (j=0;j<KK;j++) {
    x[j]=ss;                      /* bootstrap the buffer */
    ss<<=1; if (ss>=MM) ss-=MM-2; /* cyclic shift 29 bits */
  }
  x[1]++;              /* make x[1] (and only x[1]) odd */
  for (ss=seed&(MM-1),t=TT-1; t; ) {       
    for (j=KK-1;j>0;j--) x[j+j]=x[j], x[j+j-1]=0; /* "square" */
    for (j=KK+KK-2;j>=KK;j--)
      x[j-(KK-LL)]=mod_diff(x[j-(KK-LL)],x[j]),
      x[j-KK]=mod_diff(x[j-KK],x[j]);
    if (is_odd(ss)) {              /* "multiply by z" */
      for (j=KK;j>0;j--)  x[j]=x[j-1];
      x[0]=x[KK];            /* shift the buffer cyclically */
      x[LL]=mod_diff(x[LL],x[KK]);
    }
    if (ss) ss>>=1; else t--;
  }
  for (j=0;j<LL;j++) self->ran_x[j+KK-LL]=x[j];
  for (;j<KK;j++) self->ran_x[j-LL]=x[j];
  for (j=0;j<10;j++) rng_int_get_array(self,x,KK+KK-1); /* warm things up */ /* TODO: only run in _new ?*/
  self->ran_arr_ptr=(long *)&ran_arr_started;
}

long
rng_int_cycle(RngInt *self)
{
  rng_int_get_array(self, self->ran_arr_buf,QUALITY);
  self->ran_arr_buf[KK]=-1;
  self->ran_arr_ptr=self->ran_arr_buf+1;
  return self->ran_arr_buf[0];
}

long
rng_int_next(RngInt *self)
{
  return ((*self->ran_arr_ptr>=0) ? *(self->ran_arr_ptr)++ : rng_int_cycle(self));
}

