#!/bin/bash
# this does roughly what 'make distcheck' would do if we were using autotools
set -e

if ! git diff --quiet; then
    echo "You have local changes, stage them first with 'git add'!"
    exit 1
fi

eval $(grep MYPAINT_VERSION= gui/drawwindow.py)
version=$MYPAINT_VERSION
echo "Version $version"

orig=$(pwd)
d=/tmp/mypaint-$version

rm -rf $d
#svn export . $d
git checkout-index -a -f --prefix=$d/
cd $d
rm release.sh
cd ..

filename=$orig/mypaint-$version.tar.bz2
tar -cvjf $filename mypaint-$version

cd $d
scons debug=true
tests/test_mypaintlib.py
tests/test_performance.py -a -c 1
tests/test_memory_leak.py -a -e

ls -sSh $filename

echo "you can tag this release with 'git tag -s v$version'"

