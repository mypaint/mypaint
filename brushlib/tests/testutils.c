#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

#include "testutils.h"

static const char * const pass = "PASS";
static const char * const fail = "FAIL";

char *
read_file(const char *path)
{
    long file_size = -1L;
    FILE *file = fopen(path, "r");

    if (!file) {
      printf("could not open '%s'\n", path);
      perror("fopen");
      exit(1);
    }

    fseek(file , 0 , SEEK_END);
    file_size = ftell(file);
    rewind(file);

    char *buffer = (char *)malloc(sizeof(char)*file_size);
    size_t result = fread(buffer, 1, file_size, file);

    fclose(file);

    if (!result) {
        free(buffer);
        return NULL;
    }
    return buffer;
}

int
expect_int(int expected, int actual, const char *description) {
    int passed = 1;
    if (expected != actual) {
        passed = 0;
        fprintf(stderr, "%s: Expected %d, got %d\n", description, expected, actual);
    }
    return passed;
}

int
expect_float(float expected, float actual, const char *description) {
    int passed = 1;
    if (expected != actual) {
        passed = 0;
        fprintf(stderr, "%s: Expected %f, got %f\n", description, expected, actual);
    }

    return passed;
}

int
expect_true(int actual, const char *description) {
    int passed = 1;
    if (!actual) {
        passed = 0;
        fprintf(stderr, "%s: Expected %s, got %s\n", description, "TRUE", actual ? "TRUE" : "FALSE");
    }
    return passed;
}

int
test_cases_run(int argc, char **argv, TestCase *tests, int tests_n, TestCaseType type)
{
    int failures = 0;

    for (int i=0; i<tests_n; i++) {
        const TestCase *test_case = &tests[i];

        int result = test_case->function(test_case->user_data);
        if (type == TEST_CASE_NORMAL) {
            if (result != 1) {
                failures++;
            }
            fprintf(stdout, "%s: %s\n", test_case->id, (result == 1) ? pass : fail);
        } else if (type == TEST_CASE_BENCHMARK) {
            fprintf(stdout, "%s: %d ms\n", test_case->id, result);
        } else {
            assert(0);
        }
        fflush(stdout);
    }

    return (failures != 0);
}
