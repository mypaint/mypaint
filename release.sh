#!/bin/sh
set -e

version=0.3

cd ~/tmp
rm -rf mypaint-$version
svn co svn://old.homeip.net/code/mypaint mypaint-$version

cd mypaint-$version/html
./generate.py
cd ..
rm release.sh # :-)
cd ..

tar -cvjf mypaint-$version.tar.bz2 mypaint-$version
ls -sSh mypaint-$version.tar.bz2 

