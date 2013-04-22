#!/bin/bash
# 
# This script runs a couple of automated tests. It is executed by the
# jenkins build server. You can run it on your PC too. Be prepared that
# the MyPaint GUI will open full-screen and close again several times.

set -e # exit on first error
set -x # print commands before executing
cd $(dirname "$0")/..
pwd

scons -c
scons debug=1
tests/test_mypaintlib.py

# run "scons translate" commands in a sandbox checkout, so you don't end
# up with modified .po files in your working copy
rm -rf /tmp/mypaint-translate-test
git checkout-index -a -f --prefix=/tmp/mypaint-translate-test/
cd /tmp/mypaint-translate-test/
scons translate=pot
scons translate=all
cd -
rm -rf /tmp/mypaint-translate-test

rm -rf /tmp/mypaint-installtest
scons install prefix=/tmp/mypaint-installtest
rm -rf /tmp/mypaint-installtest

renice 10 $$
tests/test_performance.py -c 1 -a

# just the more lightweight memory leak tests
tests/test_memory_leak.py noleak document_alloc surface_alloc paint_save_clear
#tests/test_memory_leak.py -a

echo "Finished without error."

