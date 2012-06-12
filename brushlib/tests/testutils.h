#ifndef TESTUTILS_H
#define TESTUTILS_H

typedef int (*TestFunction) (void);

typedef struct {
    char *id;
    TestFunction function;
} TestCase;

int test_cases_run(int argc, char **argv, TestCase *tests, int tests_n);

char *read_file(char *path);

int expect_int(int expected, int actual, const char *description);
int expect_float(float expected, float actual, const char *description);
int expect_true(int actual, const char *description);

#define TEST_CASES_NUMBER(array) (sizeof(array) / sizeof(array[0]))

#endif // TESTUTILS_H
