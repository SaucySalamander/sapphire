#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include <stdio.h>
#include <stdlib.h>

/**
 * Standardized test reporting utilities for consistent output across all test suites.
 * 
 * Usage:
 *   1. Declare global counters: int tests_passed = 0, tests_failed = 0;
 *   2. Use TEST_ASSERT(condition, message) in test cases
 *   3. Call PRINT_TEST_RESULTS() at the end of main()
 */

/* Global test counters */
extern int tests_passed;
extern int tests_failed;

/**
 * Assert a test condition and record result
 */
#define TEST_ASSERT(condition, message) \
    do { \
        if (condition) { \
            printf("  ✓ %s\n", message); \
            tests_passed++; \
        } else { \
            printf("  ✗ %s\n", message); \
            tests_failed++; \
        } \
    } while (0)

/**
 * Print test section header
 */
#define TEST_SECTION(title) \
    do { \
        printf("\n============================================================\n"); \
        printf("%s\n", title); \
        printf("============================================================\n\n"); \
    } while (0)

/**
 * Print test case header
 */
#define TEST_CASE(name) \
    printf("TEST: %s\n", name)

/**
 * Print final test results summary
 */
#define PRINT_TEST_RESULTS() \
    do { \
        printf("\n============================================================\n"); \
        printf("TEST RESULTS: %d passed, %d failed\n", tests_passed, tests_failed); \
        printf("============================================================\n"); \
    } while (0)

/**
 * Print test results and return appropriate exit code
 */
#define PRINT_TEST_RESULTS_AND_EXIT() \
    do { \
        PRINT_TEST_RESULTS(); \
        return (tests_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE; \
    } while (0)

#endif /* TEST_UTILS_H */
