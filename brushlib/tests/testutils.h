#ifndef TESTUTILS_H
#define TESTUTILS_H

typedef int (*TestFunction) (void *user_data);

typedef struct {
    char *id;
    TestFunction function;
    void *user_data;
} TestCase;

typedef enum {
    TEST_CASE_NORMAL = 0,
    TEST_CASE_BENCHMARK
} TestCaseType;

int test_cases_run(int argc, char **argv, TestCase *tests, int tests_n, TestCaseType type);

char *read_file(const char *path);

int expect_int(int expected, int actual, const char *description);
int expect_float(float expected, float actual, const char *description);
int expect_true(int actual, const char *description);

#define TEST_CASES_NUMBER(array) (sizeof(array) / sizeof(array[0]))

#endif // TESTUTILS_H
