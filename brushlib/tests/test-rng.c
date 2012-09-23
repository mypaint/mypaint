#include "rng-double.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "testutils.h"

int
test_rng_double_smoke(void *user_data)
{
    register int m; double a[2009];

    RngDouble* gen = rng_double_new(310952L);
    for (m=0;m<2009;m++) rng_double_get_array(gen, a, 1009);
    printf("%.20f\n", a[0]);

    assert(a[0] == 0.13283300318196644696);

    // XXX: possible issue in original impl:
    // in the original test from D.Knuth, the internal state (ran_u) was tested
    // directly and the result was then the same for both of these cases
    // assert(ran_u[0] == 0.36410514377569680455);
    // in the int version, a is tested like we do here and also gives same result for the two cases
    // this is not the case here though?

    rng_double_free(gen);

    gen = rng_double_new(310952L);
    for (m=0;m<1009;m++) rng_double_get_array(gen, a, 2009);
    printf("%.20f\n", a[0]);

    assert(a[0] == 0.64694669426788964373);

    rng_double_free(gen);

    return 1;
}

int
main(int argc, char **argv)
{
    TestCase test_cases[] = {
        {"/rng/double/smoke", test_rng_double_smoke, NULL}
    };

    return test_cases_run(argc, argv, test_cases, TEST_CASES_NUMBER(test_cases), 0);
}
