
import os, sys

tests_dir = os.path.join(os.path.dirname(__file__), '../tests')
tests_dir = os.path.abspath(tests_dir)
sys.path.insert(0, tests_dir)
import ctests

def is_cbenchmark(fn):
    return fn.startswith('benchmark-') and not os.path.splitext(fn)[1]

def test_benchmark_brushlib():
    c_benchmarks = [os.path.abspath(os.path.join(tests_dir, fn)) for fn in os.listdir(tests_dir) if is_cbenchmark(fn)]

    for executable in c_benchmarks:
        yield ctests.run_ctest, executable
