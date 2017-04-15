#!/usr/bin/env bash
# MSYS2 build and test commands.
# All rights waived: https://creativecommons.org/publicdomain/zero/1.0/
#
# This script is designed to be called by AppVeyor or Tea-CI.
#
# XXX: THIS SCRIPT WILL MESS WITH YOUR SYSTEM BY INSTALLING STUFF
#
# It expects to be called with MSYSTEM="MINGW{64,32}", i.e. from an
# MSYS2 "native" shell.


set -x
set -e

if test "x$CI" = "x" || ! "$CI"; then
    echo "Not in a CI environment. Not running."
    exit 1
fi

SCRIPT=`basename "$0"`
SCRIPTDIR=`dirname "$0"`
TOPDIR=`dirname "$SCRIPTDIR"`

cd "$TOPDIR"

case "$MSYSTEM" in
    "MINGW64")
        PKG_PREFIX="mingw-w64-x86_64"
        MINGW_INSTALLS="mingw64"
        ;;
    "MINGW32")
        PKG_PREFIX="mingw-w64-i686"
        MINGW_INSTALLS="mingw32"
        ;;
    *)
        echo >&2 "$SCRIPT must only be called from a MINGW64/32 login shell."
        exit 1
        ;;
esac
export MINGW_INSTALLS

PACMAN_SYNC="pacman -S --noconfirm --needed --noprogressbar"


install_dependencies() {
    # Try to solve potential conflicts up front, for AppVeyor.
    pacman --remove --noconfirm repman-git || true
    pacman --remove --noconfirm libmypaint-git || true
    pacman --remove --noconfirm libmypaint || true
    # Pre-built ones
    $PACMAN_SYNC \
        ${PKG_PREFIX}-toolchain \
        ${PKG_PREFIX}-pkg-config \
        ${PKG_PREFIX}-glib2 \
        ${PKG_PREFIX}-gtk3 \
        ${PKG_PREFIX}-json-c \
        ${PKG_PREFIX}-lcms2 \
        ${PKG_PREFIX}-python2-cairo \
        ${PKG_PREFIX}-pygobject-devel \
        ${PKG_PREFIX}-python2-gobject \
        ${PKG_PREFIX}-python2-numpy \
        ${PKG_PREFIX}-hicolor-icon-theme \
        ${PKG_PREFIX}-librsvg \
        ${PKG_PREFIX}-gobject-introspection \
        ${PKG_PREFIX}-python2-nose \
        base-devel git scons

    # We may be building a user's clone of mypaint from github during a
    # CI check. If such a thing exists, try a raw install first.
    uri1=`git config --get remote.origin.url`
    uri2=`echo "$uri1" | sed 's~/mypaint\([.]git\)\?$~/libmypaint\1~'`

    builddir="/tmp/build.$$.libmypaint"
    rm -fr "$builddir"
    if git clone "$uri2" "$builddir"; then
        cd "$builddir"
        ./autogen.sh
        ./configure --enable-shared --enable-static
        make
        make -j1 install
        cd $TOPDIR
        return
    fi

    # Alexey is building libmypaint now, so use that as a fallback.
    $PACMAN_SYNC ${PKG_PREFIX}-libmypaint
}


# Convienience aliases for SCons stuff.

build_for_testing() {
    scons
}


clean_local_repo() {
    scons --clean
}

# Can't test everything from AppVeyor, nor can we use the executable bit
# on Windows to discern which ones it's currently sensible to run.
# However it's always appropriate to run the doctests.

run_tests() {
    nosetests-2.7 --with-doctest lib/*.py lib/*/*.py
}


# Command line processing

case "$1" in
    installdeps)
        install_dependencies
        ;;
    build)
        build_for_testing
        ;;
    clean)
        clean_local_repo
        ;;
    test)
        run_tests
        ;;
    *)
        echo >&2 "usage: $SCRIPT {installdeps|build|test|clean}"
        exit 2
        ;;
esac
