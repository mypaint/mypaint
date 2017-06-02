#!/usr/bin/env bash
# MSYS2 build and test script for MyPaint.
# All rights waived: https://creativecommons.org/publicdomain/zero/1.0/
#
#: Usage:
#:   $ msys2_build.sh [OPTIONS]
#:
#: OPTIONS:
#:   installdeps    Install dependencies MyPaint requires
#:   build          Build MyPaint from Source
#:   clean          Clean the Build tree.
#:   tests          Run test on build. Run "build" first.
#:   doctest        Checks to make sure all nessesary build files are present
#:   bundle         Creates an bundle installer for Windows' builds.
#
# This script is initially designed to be called by AppVeyor or Tea-CI.
# However it's clean enough to run from an interactive shell. It expects
# to be called with MSYSTEM="MINGW{64,32}", i.e. from an MSYS2 "native"
# shell.

set -e
set -x

GREEN='\033[0;32m'
NC='\033[0m' # No Color

SCRIPT=`basename "$0"`
SCRIPTDIR=`dirname "$0"`
TOPDIR=`dirname "$SCRIPTDIR"`

cd "$TOPDIR"

case "$MSYSTEM" in
    "MINGW64")
        PKG_PREFIX="mingw-w64-x86_64"
        MINGW_INSTALLS="mingw64"
        BITS=32
        ;;
    "MINGW32")
        PKG_PREFIX="mingw-w64-i686"
        MINGW_INSTALLS="mingw32"
        BITS=64
        ;;
    *)
        echo >&2 "$SCRIPT must only be called from a MINGW64/32 login shell."
        exit 1
        ;;
esac
export MINGW_INSTALLS

# For more information on how to use pacman visit:
# https://www.archlinux.org/pacman/pacman.8.html or
# use "man pacman" in your terminal if the package is installed.
PACMAN_SYNC="pacman -S --noconfirm --needed --noprogressbar"

# These are URI links to the PKGBUILD files.
# https://github.com/Alexpux/MINGW-packages
LIBMYPAINT_PKGBUILD_URI="https://raw.githubusercontent.com/Alexpux/MINGW-packages/master/mingw-w64-libmypaint-git/PKGBUILD"

# Location of built libmypaint package
PKG_DIR="/tmp/pkg.mingw"

# Ouput location of where the build and working directories
# are located in Appveyor.
OUTPUT_DIR="/tmp/mypaint-builds"
TARGET_DIR="${TOPDIR}${OUTPUT_DIR}"

install_dependencies(){
    # Try to solve potential conflicts up front, for AppVeyor.
    echo -e "${GREEN}+++BASH: Installing Dependencies of MyPaint${NC}"
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
        ${PKG_PREFIX}-python2-setuptools \
        swig \
        base-devel \
        git
}

create_pkg-dir(){
    #Creates a temporary directory for libmypaint's or other PKGBUILD packages.
    rm -rf "$PKG_DIR"
    mkdir -p "$PKG_DIR"
    cd $TOPDIR
}

build_libmypaint(){
    echo -e "${GREEN}+++BASH: Building Libmypaint MINGW PKGBUILD.${NC}"
    BUILD_DIR="/tmp/build.libmypaint"
    rm -rf "$BUILD_DIR"
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    curl --remote-name "$LIBMYPAINT_PKGBUILD_URI"
    MSYSTEM="MSYS2" bash --login -c "cd $BUILD_DIR && makepkg-mingw -f"
    # List Package to make sure it is built and ready.
    ls -la *.pkg.tar.xz
    # Move Package to tmp package folder so it can be used for bundle.
    mv *.pkg.tar.xz "$PKG_DIR"
    cd $TOPDIR
    rm -rf "$BUILD_DIR"
}

install_libmypaint(){
    # You need to run build_libmypaint first before doing this.
    echo -e "${GREEN}+++BASH: Installing Libmypaint from MINGW package.${NC}"
    cd "$PKG_DIR"
    # Double check to make sure the package is there.
    ls -la *.pkg.tar.xz
    pacman -U --noconfirm *.pkg.tar.xz
    cd $TOPDIR
}

bundle_mypaint(){
    # Technically we don't need to run tests since it was already done.
    # Besides those test areas need to be updated with setuptools in mind.
    echo -e "${GREEN}+++BASH: Bundling MyPaint for Windows.${NC}"
    windows/build.sh --sloppy --skip-deps --extra-pkgs "$PKG_DIR"
}

# Test Build, Clean, and Install tools to make sure all of setup.py is
# working as intended.
build_for_testing() {
    echo -e "${GREEN}+++BASH: Building MyPaint from Source.${NC}"
    python setup.py build
}

clean_local_repo() {
    echo -e "${GREEN}+++BASH: Cleaning Local Build.${NC}"
    python setup.py clean --all
    rm -vf lib/*_wrap.c*
}

install_test(){
    # TODO: Look into this to find out why it is failing.
    echo -e "${GREEN}+++BASH: Testing Setup.py Managed Instalation Scripts.${NC}"
    python setup.py managed_install
    python setup.py managed_uninstall
}

# Can't test everything from TeaCI due to wine crashing.
# However, it's always appropriate to run the doctests.
# With Appveyor, the tests scripts should run just fine.
run_doctest() {
    echo -e "${GREEN}+++BASH: Running Doc Tests.${NC}"
    python setup.py nosetests --tests lib
}

run_tests(){
    echo -e "${GREEN}+++BASH: Running Test Suite.${NC}"
    python setup.py test
}

copy_builds(){
    # This is required in order to upload to artifacts in Appveyor.
    # https://www.appveyor.com/docs/packaging-artifacts/
    rm -f $TARGET_DIR
    mkdir -p $TARGET_DIR
    echo -e "${GREEN}+++BASH: Copying installer to ${TARGET_DIR}.${NC}"
    cp -a $OUTPUT_DIR/*.exe "${TARGET_DIR}"
    echo -e "${GREEN}+++BASH: Copying zip file to ${TARGET_DIR}.${NC}"
    cp -a $OUTPUT_DIR/*.zip "${TARGET_DIR}"
    echo -e "${GREEN}+++BASH: All Files Copied.${NC}"
}

# Command line processing

case "$1" in
    installdeps)
        install_dependencies
        create_pkg-dir
        build_libmypaint
        install_libmypaint
        ;;
    build)
        build_for_testing
        ;;
    clean)
        clean_local_repo
        ;;
    tests)
        run_tests
#        install_test
        ;;
    doctest)
        run_doctest
        ;;
    bundle)
        bundle_mypaint
        copy_builds
        ;;
    *)
        echo >&2 "usage: $SCRIPT {installdeps|build|clean|tests|doctest|bundle}"
        exit 2
        ;;
esac
