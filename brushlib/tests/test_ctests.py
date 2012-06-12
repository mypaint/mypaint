
import os

"""Simple nosetest compatible test wrapper that runs the plain-C tests found in brushlib."""

# TODO: get more fine grained test setup
# * Make the C test lib be able to report the registered testcases
# * Make the C test lib be able to run a single specified test
# * Use this to generate and execute one test per case

tests_dir = os.path.dirname(os.path.abspath(__file__))
lib_dir = os.path.abspath(os.path.join(tests_dir, '../..'))

def is_ctest(fn):
    return fn.startswith('test-') and not os.path.splitext(fn)[1]

def test_brushlib():
    c_tests = [os.path.abspath(os.path.join(tests_dir, fn)) for fn in os.listdir(tests_dir) if is_ctest(fn)]

    for executable in c_tests:
        yield run_ctest, executable

def run_ctest(executable):
    import subprocess

    environ = {}
    environ.update(os.environ)
    environ.update({'LD_LIBRARY_PATH': lib_dir})

    retval = subprocess.call(executable, env=environ, cwd=tests_dir)
    assert (retval == 0)
