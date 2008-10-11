#!/bin/sh
# Run this to generate all the initial makefiles, etc.

echo "autotools build is currently unmaintained, please use SConstruct"
exit 1

#aclocal -I macros && libtoolize --copy && autoheader && autoconf && automake --add-missing --copy
aclocal -I macros && libtoolize --copy && autoheader && autoconf && automake --foreign --add-missing --copy

