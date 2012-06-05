
#include "rng-int.h"
#include "rng-double.h"

#include <stdio.h>
#include <malloc.h>
#include <assert.h>

#include "testutils.h"

int
test_rng_int_smoke()
{
    register int m; long a[2009];

    RngInt* gen = rng_int_new(310952L);
    for (m=0;m<=2009;m++) rng_int_get_array(gen, a,1009);
    printf("%ld\n", a[0]);

    assert(a[0] == 995235265);

    rng_int_free(gen);

    gen = rng_int_new(310952L);
    for (m=0;m<=1009;m++) rng_int_get_array(gen, a,2009);
    printf("%ld\n", a[0]);

    assert(a[0] == 995235265);

    rng_int_free(gen);

    return 1;
}

int
test_rng_double_smoke()
{
    register int m; double a[2009];

    RngDouble* gen = rng_double_new(310952L);
    for (m=0;m<2009;m++) rng_double_get_array(gen, a, 1009);
    printf("%.20f\n", a[0]);

    assert(a[0] == 0.40565695769206500110);

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

    assert(a[0] == 0.04731923453148767500);

    rng_double_free(gen);

    return 1;
}

int
main(int argc, char **argv)
{
    TestCase test_cases[] = {
        {"/rng/int/smoke", test_rng_int_smoke},
        {"/rng/double/smoke", test_rng_double_smoke}
    };

    return test_cases_run(argc, argv, test_cases, TEST_CASES_NUMBER(test_cases));
}
