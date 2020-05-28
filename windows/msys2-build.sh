#!/usr/bin/env bash
# MSYS2 build and test script for MyPaint.
# All rights waived: https://creativecommons.org/publicdomain/zero/1.0/
#
#: Usage:
#:   $ msys2_build.sh [OPTIONS]
#:
#: OPTIONS:
#:   installdeps  Build+install dependencies.
#:   build        Build MyPaint itself from this source tree.
#:   clean        Clean the build tree.
#:   tests        Runs tests on the built source.
#:   doctest      Check to make sure all python docs work.
#:   bundle       Creates installer bundles in ./out/bundles
#:
#:  This script is designed to be called by AppVeyor or Tea-CI. However
#:  it's clean enough to run from an interactive shell. It expects to be
#:  called with MSYSTEM="MINGW{64,32}", i.e. from an MSYS2 "native" shell.
#:
#:  Build artifacts are written to ./out/pkgs and ./out/bundles by default.

set -e

# ANSI control codes
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Script name and location.
SCRIPT=`basename "$0"`
SCRIPTDIR=`dirname "$0"`
cd "$SCRIPTDIR/.."

# Main repository location, as an absolute path.
TOPDIR=`pwd`
cd "$TOPDIR"

# Ensure we're being run from one of MSYS2's "native shells".
case "$MSYSTEM" in
    "MINGW64")
        PKG_PREFIX="mingw-w64-x86_64"
        MINGW_INSTALLS="mingw64"
        BUNDLE_ARCH="w64"
        ;;
    "MINGW32")
        PKG_PREFIX="mingw-w64-i686"
        MINGW_INSTALLS="mingw32"
        BUNDLE_ARCH="w32"
        ;;
    *)
        echo >&2 "$SCRIPT must only be called from a MINGW64/32 login shell."
        exit 1
        ;;
esac
export MINGW_INSTALLS

# This script pulls down and maintains a clone of the pkgbuild tree for
# MSYS2's MINGW32 and MINGW64 software.
SRC_ROOT="${SRC_ROOT:-/tmp/src}"
SRC_PROJECT="mingw"
SRC_DIR="${SRC_ROOT}/${SRC_PROJECT}"
SRC_CLONEURI="https://github.com/Alexpux/MINGW-packages.git"

# Output location for build artifacts.
OUTPUT_ROOT="${OUTPUT_ROOT:-$TOPDIR/out}"

install_dependencies() {

    loginfo "Removing potential package conflicts..."
    pacman --remove --noconfirm ${PKG_PREFIX}-mypaint || true
    pacman --remove --noconfirm ${PKG_PREFIX}-mypaint || true
    pacman --remove --noconfirm ${PKG_PREFIX}-libmypaint || true
    pacman --remove --noconfirm ${PKG_PREFIX}-mypaint-brushes2 || true

    loginfo "Installing pre-built dependencies for MyPaint"
    pacman -S --noconfirm --needed --noprogressbar \
        ${PKG_PREFIX}-toolchain \
        ${PKG_PREFIX}-pkg-config \
        ${PKG_PREFIX}-glib2 \
        ${PKG_PREFIX}-gtk3 \
        ${PKG_PREFIX}-json-c \
        ${PKG_PREFIX}-lcms2 \
        ${PKG_PREFIX}-python3-cairo \
        ${PKG_PREFIX}-pygobject-devel \
        ${PKG_PREFIX}-python3-gobject \
        ${PKG_PREFIX}-python3-numpy \
        ${PKG_PREFIX}-hicolor-icon-theme \
        ${PKG_PREFIX}-librsvg \
        ${PKG_PREFIX}-gobject-introspection \
        ${PKG_PREFIX}-python3-nose \
        ${PKG_PREFIX}-python3-setuptools \
        ${PKG_PREFIX}-swig \
        ${PKG_PREFIX}-gsettings-desktop-schemas \
        base-devel \
        git

    loginfo "Installing pre-built dependencies for Styrene + its install"
    pacman -S --noconfirm --needed --noprogressbar \
        ${PKG_PREFIX}-nsis \
        ${PKG_PREFIX}-gcc \
        ${PKG_PREFIX}-binutils \
        ${PKG_PREFIX}-python3 \
        ${PKG_PREFIX}-python3-pip \
        zip \
        p7zip

    logok "Dependencies installed."

    gdk-pixbuf-query-loaders --update-cache
    logok "Pixbuf query loaders cache updated"
}


loginfo() {
    echo -ne "${CYAN}"
    echo -n "$@"
    echo -e "${NC}"
}


logok() {
    echo -ne "${GREEN}"
    echo -n "$@"
    echo -e "${NC}"
}


logerr() {
    echo -ne "${RED}ERROR: "
    echo -n "$@"
    echo -e "${NC}"
}


check_output_dir() {
    type="$1"
    if test -d "$OUTPUT_ROOT/$type"; then
        return
    fi
    mkdir -vp "$OUTPUT_ROOT/$type"
}


update_mingw_src() {
    # Initialize or update the managed MINGW-packages sources dir.
    if test -d "$SRC_DIR"; then
        loginfo "Updating $SRC_DIR..."
        pushd "$SRC_DIR"
        git pull
    else
        loginfo "Creating $SRC_ROOT"
        mkdir -vp "$SRC_ROOT"
        pushd "$SRC_ROOT"
        loginfo "Shallow-cloning $SRC_CLONEURI into $SRC_DIR..."
        git clone --depth 1 "$SRC_CLONEURI" "$SRC_PROJECT"
    fi
    popd
    logok "Updated $SRC_DIR"
}


seed_mingw_src_mypaint_repo() {
    # Seed the MyPaint source repository that makepkg-mingw wants
    # from this one if it doesn't yet exist.
    # The mypaint repo is quite big, so let's save some bandwidth!
    repo="$SRC_DIR/mingw-w64-mypaint/mypaint"
    test -d "$TOPDIR/.git" || return
    test -d "$repo" && return
    loginfo "Seeding $repo..."
    git clone --local --no-hardlinks --bare "$TOPDIR" "$repo"
    pushd "$repo"
    git remote remove origin
    git remote add origin https://github.com/mypaint/mypaint.git
    git fetch --tags origin
    popd
    logok "Seeded $repo"
}


build_pkg() {
    # Build and optionally install a .pkg.tar.zst from the
    # managed tree of PKGBUILDs.
    #
    # Usage: build_pkg PKGNAMESTEM {true|false}

    if ! test -d "$SRC_DIR"; then
        logerr "Managed src dir $SRC_DIR does not exist (update_mingw_src 1st)"
        exit 2
    fi

    pkgstem="$1"
    install="$2"
    src="${SRC_DIR}/mingw-w64-$pkgstem"
    pushd "$src"
    rm -vf *.pkg.tar.zst

    # This only builds for the arch in MINGW_INSTALLS, i.e. the current
    # value of MSYSTEM.
    loginfo "Building in $src for $MINGW_INSTALLS ..."
    MSYSTEM=MSYS2 bash --login -c 'cd "$1" && makepkg-mingw -f' - "$src"
    logok "Build finished."

    if $install; then
        loginfo "Installing built packages..."
        pacman -U --noconfirm *.pkg.tar.zst
        logok "Install finished."
    fi
    popd

    loginfo "Capturing build artifacts..."
    check_output_dir "pkgs"
    mv -v "$src"/*.pkg.tar.zst "$OUTPUT_ROOT/pkgs"
    logok "Packages moved."
}


bundle_mypaint() {
    # Convert local and repository *.pkg.tar.zst into nice bundles
    # for users to install.
    # Needs the libmypaint and mypaint .pkg.tar.zst artifacts.
    styrene_path=`which styrene||true`
    if [ "x$styrene_path" = "x" ]; then
        mkdir -vp "$SRC_ROOT"
        pushd "$SRC_ROOT"
        if [ -d styrene ]; then
            loginfo "Updating managed Styrene source"
            pushd styrene
            git pull
        else
            loginfo "Cloning managed Styrene source"
            git clone https://github.com/mypaint/mypaint-styrene.git styrene
            pushd styrene
        fi
        loginfo "Installing styrene with pip3..."
        pip3 install .
        loginfo "Installed styrene."
        popd
        popd
    fi

    check_output_dir "bundles"
    loginfo "Creating installer bundles..."

    tmpdir="/tmp/styrene.$$"
    mkdir -p "$tmpdir"
    styrene --colour=yes \
            --no-zip \
            --7z \
            --pkg-dir="$OUTPUT_ROOT/pkgs" \
            --output-dir="$tmpdir" \
            "$TOPDIR/windows/styrene/mypaint.cfg"

    output_version=$(echo $BUNDLE_ARCH-$APPVEYOR_BUILD_VERSION | sed -e 's/[^a-zA-Z0-9._-]/-/g')

    mv -v "$tmpdir"/*-standalone.7z \
        "$OUTPUT_ROOT/bundles/mypaint-$output_version-standalone.7z"
    mv -v "$tmpdir"/*-installer.exe  \
        "$OUTPUT_ROOT/bundles/mypaint-$output_version-installer.exe"

    ls -l "$OUTPUT_ROOT/bundles"/*.*

    rm -fr "$tmpdir"

    logok "Bundle creation finished."
}

# Test Build, Clean, and Install tools to make sure all of setup.py is
# working as intended.

build_for_testing() {
    loginfo "Building MyPaint from source"
    python3 setup.py build
    logok "Build finished."
}

clean_local_repo() {
    loginfo "Cleaning local build"
    python3 setup.py clean --all
    rm -vf lib/*_wrap.c*
    logok "Clean finished."
}

install_test(){
    # TODO: Look into this to find out why it is failing.
    loginfo "Testing setup.py managed installation commands"
    python3 setup.py managed_install
    python3 setup.py managed_uninstall
    logok "Install-test finished finished."
}

# Can't test everything from TeaCI due to wine crashing.
# However, it's always appropriate to run the doctests.
# With Appveyor, the tests scripts should run just fine.

run_doctest() {
    loginfo "Running unit tests."
    python3 setup.py nosetests --tests lib
    logok "Unit tests done."
}

run_tests() {
    loginfo "Runnning startup/shutdown test"
    MYPAINT_DEBUG=1 python setup.py demo --args='--run-and-quit'
    loginfo "Running conformance tests."
    python3 setup.py test
    logok "Tests done."
}


# Command line processing

case "$1" in
    installdeps)
        install_dependencies
        update_mingw_src
        src="${SRC_DIR}/mingw-w64-libmypaint"
        cp ./windows/PKGBUILD-libmypaint $src/PKGBUILD
        build_pkg "libmypaint" true
        src="${SRC_DIR}/mingw-w64-mypaint-brushes2"
        cp ./windows/PKGBUILD-mypaint-brushes2 $src/PKGBUILD
        build_pkg "mypaint-brushes2" true
        ;;
    build)
        build_for_testing
        ;;
    clean)
        clean_local_repo
        ;;
    tests)
        run_tests
        # install_test
        ;;
    doctest)
        run_doctest
        ;;
    bundle)
        update_mingw_src
        seed_mingw_src_mypaint_repo
        src="${SRC_DIR}/mingw-w64-mypaint"
        cp ./windows/PKGBUILD-mypaint $src/PKGBUILD
        build_pkg "mypaint" false
        bundle_mypaint
        ;;
    *)
        grep '^#:' $0 | cut -d ':' -f 2-50
        exit 2
        ;;
esac
