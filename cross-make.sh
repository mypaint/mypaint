#!/bin/sh

# copied from http://www.libsdl.org/extras/win32/cross/README.txt

PREFIX=/usr/local/cross-tools
TARGET=i386-mingw32msvc
PATH="$PREFIX/bin:$PREFIX/$TARGET/bin:$PATH"
export PATH
exec make $*
