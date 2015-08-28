#!/bin/sh
# This script runs a number of automated tests.
# It is executed routinely by the Travis build server.
# You can run it on your PC too.
#
# Be prepared that the MyPaint GUI will open full-screen and close again
# several times if you are running this code on a system with a
# graphical (X11) display.

set -e # exit on first error
set -x # print commands before executing
cd $(dirname "$0")/..
pwd

# Headless correctness tests
scons -c
scons debug=1
tests/test_compositeops.py

# Slightly more performance-oriented tests
scons -c
scons
tests/test_mypaintlib.py
tests/test_brushlib.py

# run "scons translate" commands in a sandbox, so you don't end
# up with modified .po files in your working copy
rm -rf /tmp/mypaint-translate-test
(
    cp -a . /tmp/mypaint-translate-test
    cd /tmp/mypaint-translate-test/
    scons translate=pot
    scons translate=all
)
rm -rf /tmp/mypaint-translate-test

# Test installing in the basic way
rm -rf /tmp/mypaint-installtest
scons install prefix=/tmp/mypaint-installtest
rm -rf /tmp/mypaint-installtest

# GUI performance testing - only if there is a $DISPLAY
if test "x$DISPLAY" != "x"; then
    renice 10 $$
    tests/test_rendering.py
    tests/test_performance.py -c 1 -a
fi

# just the more lightweight memory leak tests
#tests/test_memory_leak.py noleak document_alloc surface_alloc paint_save_clear

#tests/test_memory_leak.py -a

echo "Finished without error."
