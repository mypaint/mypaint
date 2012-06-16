
import os
import ctests

"""Simple nosetest compatible test wrapper that runs the plain-C tests found in brushlib."""

tests_dir = ctests.tests_dir

def is_ctest(fn):
    return fn.startswith('test-') and not os.path.splitext(fn)[1]

def test_brushlib():
    c_tests = [os.path.abspath(os.path.join(tests_dir, fn)) for fn in os.listdir(tests_dir) if is_ctest(fn)]

    for executable in c_tests:
        yield ctests.run_ctest, executable

