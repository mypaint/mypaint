#!/bin/sh
rm -rf /tmp/mypaint-distcheck
svn export . /tmp/mypaint-distcheck
cd /tmp/mypaint-distcheck
scons -j2 && ./mypaint

