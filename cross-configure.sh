#!/bin/sh

# adapted from http://www.libsdl.org/extras/win32/cross/README.txt

CONFIG_SHELL=/bin/sh
export CONFIG_SHELL
TARGET=i586-mingw32msvc
CROSS_DEV=$HOME/Win32/mypaint_win32
DLLZIP=$HOME/MyPaintDLL.zip
PATH="$PATH:$CROSS_DEV/bin"
export PATH

#PYTHON is copied from a windows install (with pygtk, pycairo, pysqlite) into:
# $CROSS_DEV/Python24/

#########################################
# Libs using pkg-config
#
# rem: for me libxml2 as and hand-written pc file
#
# PKG_CONFIG in cross-configuration.
PKG_CONFIG_LIBDIR="$CROSS_DEV/lib/pkgconfig:$CROSS_DEV/Python24/Lib/pkgconfig"

DIRS="$(ls  $CROSS_DEV)"

# all my libs are in $CROSS_DEV/__libname__/
# the *.pc files needs there prefix adapted
for f in $DIRS; do
  if test -d $CROSS_DEV/$f/lib/pkgconfig ; then
     PKG_CONFIG_LIBDIR="$PKG_CONFIG_LIBDIR:$CROSS_DEV/$f/lib/pkgconfig" ;
  fi
done
# /PKG_CONFIG
#########################################


#########################################
# PYTHON
export PKG_CONFIG_LIBDIR
export PKG_CONFIG_PATH
PYTHONHOME="$CROSS_DEV/Python24"
PYTHONPATH="$CROSS_DEV/Python24/Lib"
PYTHON_INCLUDES="-I$CROSS_DEV/Python24/include"
PYTHON_LIBS="-L$CROSS_DEV/Python24/libs -lpython24"

# GCompris configure will skip PYTHON_TEST if this is exported:
export PYTHONHOME
export PYTHONPATH
export PYTHON_INCLUDES
export PYTHON_LIBS
# /PYTHON
##########################################
#libiconv
CFLAGS="$CFLAGS -I$CROSS_DEV/libiconv/include"
LDFLAGS="$LDFLAGS -L$CROSS_DEV/libiconv/lib"

#gettext
CFLAGS="$CFLAGS -I$CROSS_DEV/gettext/include"
LDFLAGS="$LDFLAGS -L$CROSS_DEV/gettext/lib"

#zlib
#CFLAGS="$CFLAGS -I$CROSS_DEV/zlib/include"
#LDFLAGS="$LDFLAGS -L$CROSS_DEV/zlib/lib"

# LIBXML2
# !!!!!! libxml2 if it does not hace pc file
# adapt CFLAGS
# adapt LDFLAGS

############################################

cache=cross-config.cache
CFLAGS="$CFLAGS"  LDFLAGS="$LDFLAGS" sh configure --cache-file="$cache" \
	--target=$TARGET --host=$TARGET --build=i386-linux \
         --enable-nsis --with-dllzip=$DLLZIP
        $*

status=$?
rm -f "$cache"
exit $status
