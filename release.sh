#!/bin/sh
set -e

FIXME: outdated

version=0.4.1

cd ~/tmp
rm -rf mypaint-$version
svn export svn://old.homeip.net/code/mypaint mypaint-$version

cd mypaint-$version/html
./generate.py
rm *.pyc
cd ..
rm release.sh # :-)
cd ..

tar -cvjf mypaint-$version.tar.bz2 mypaint-$version
ls -sSh mypaint-$version.tar.bz2 

