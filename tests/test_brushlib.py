import sys, os

p = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, p)
sys.path.insert(0, os.path.join(p, 'lib'))

# Wrapper script/executable for running the brushlib C tests
# against the Python-based tile surface

from lib import mypaintlib
sys.exit(mypaintlib.run_brushlib_tests())
