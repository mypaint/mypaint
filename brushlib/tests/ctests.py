
import os

# TODO: get more fine grained test setup
# * Make the C test lib be able to report the registered testcases
# * Make the C test lib be able to run a single specified test
# * Use this to generate and execute one test per case

tests_dir = os.path.dirname(os.path.abspath(__file__))
lib_dir = os.path.abspath(os.path.join(tests_dir, '../..'))

def run_ctest(executable):
    import subprocess

    environ = {}
    environ.update(os.environ)
    ld_library_path = environ.get('LD_LIBRARY_PATH', '')
    ld_library_path += ':%s' % lib_dir
    environ.update({'LD_LIBRARY_PATH': ld_library_path})

    retval = subprocess.call(executable, env=environ, cwd=tests_dir)
    assert (retval == 0)
