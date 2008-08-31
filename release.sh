#!/bin/sh
# this does roughly what 'make distcheck' would do if it did work
set -e

version=0.5.1

orig=$(pwd)
d=/tmp/mypaint-$version

rm -rf $d
svn export . $d
cd $d/html
./generate.py
rm *.pyc
cd ..
rm release.sh
rpl "SVNVERSION=" "SVNVERSION=$version #" configure.in
if ! grep "MYPAINT_VERSION='$version'"  drawwindow.py ; then
    rpl "MYPAINT_VERSION=" "MYPAINT_VERSION='$version' #" drawwindow.py
fi
./autogen.sh
cd ..

filename=$orig/mypaint-$version.tar.bz2
tar -cvjf $filename mypaint-$version

cd $d
./configure
make

ls -sSh $filename
