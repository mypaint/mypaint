#!/usr/bin/env python2
# Wrapper script/executable for running the brushlib C tests
# against the Python-based tile surface.

import sys
import os

# The C code ultimately imports lib.brushlib, but we need to chdir so
# that it can find its test data.
p = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, p)

from lib import mypaintlib
os.chdir("brushlib/tests")
sys.exit(mypaintlib.run_brushlib_tests())
