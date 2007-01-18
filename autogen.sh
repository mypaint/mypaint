#!/bin/sh
# Run this to generate all the initial makefiles, etc.

#aclocal -I macros && libtoolize --copy && autoheader && autoconf && automake --add-missing --copy
aclocal -I macros && libtoolize --copy && autoheader && autoconf && automake --foreign --add-missing --copy

