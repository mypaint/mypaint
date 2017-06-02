#!/bin/sh
#: Build binary distributions of MyPaint for Windows from git.
#:
#: Usage:
#:   $ windows/build.sh [OPTIONS] [--] [RELEASETARBALL]
#:
#: Options:
#:   --help         show this message and exit ok
#:   --sloppy       skip tests, no cleanup, reuse existing target areas
#:   --skip-deps    skips installation of Dependencies. Use if building from Appveyor.
#:   --show-output  open output folder in Windows if build succeeded
#:   --extra-pkgs DIR  folder with extra/replacement packages for target
#:
#: Rigourous builds (--sloppy flag not specified) are our standard for
#: release, and are the default, but the amount of reinstallation and
#: rebuilding they involve can be tedious for repeated testing.
#:
#: If you have Inno Setup installed, its ISCC.exe will be called with an
#: auto-generated .iss script to make a user-friendly setup.exe in
#: addition to the standalone .zip.
#:
#: Final output is written into a temp folder, and if the --showoutput
#: flag is specified on the command line, Windows Explorer is launched
#: on it so you can copy the .zip and .exe files somewhere else.
#:
#: You MUST install MSYS2 <https://msys2.github.io/> before running this
#: script, and it MUST be run from either the MINGW32 or MINGW64 shell.
#:
#: If no RELEASETARBALL is specified, one will be created for you with
#: the ../release.sh script.
#:
#: The extra-pkgs folder contains binary .pkg.tar.xz files which will
#: be installed into the target tree after the regular dependencies
#: and before pruning the tree starts. This is intended as a way of
#: incorporating fixes which only exist in local builds
#: of dependency packages.

set -e
set -x

GREEN='\033[0;32m'
NC='\033[0m' # No Color

RIGOUROUS=true
SHOW_OUTPUT=false
SKIP_DEPENDENCIES=false
RELEASE_TARBALL=
EXTRA_PACKAGES_DIR=

while test $# -gt 0; do
    case "$1" in
        --help)
            grep '^#:' $0
            exit 0
            ;;
        --sloppy)
            RIGOUROUS=false
            shift
            ;;
        --show-output)
            SHOW_OUTPUT=true
            shift
            ;;
        --skip-deps)
            SKIP_DEPENDENCIES=true
            shift
            ;;
        --extra-pkgs)
            shift
            EXTRA_PACKAGES_DIR="$1"
            shift
            ;;
        --)
            shift
            break
            ;;
        -*)
            echo >&2 "Unknown option $1 (try running with --help)"
            exit 1
            ;;
        *)
            break
            ;;
    esac
done
if test $# -gt 0; then
    RELEASE_TARBALL="$1"
    shift
fi
if test $# -gt 0; then
    echo >&2 "Trailing junk in args: \"$@\" (try running with --help)"
    exit 1
fi


# Pre-flight checks. The MSYSTEM architecture (and separately, the git
# export revision) are used to distinguish one build area from another.

case "x$MSYSTEM" in
    xMINGW32)
        PKG_PREFIX="mingw-w64-i686"
        ARCH="i686"
        BITS=32
        ;;
    xMINGW64)
        PKG_PREFIX="mingw-w64-x86_64"
        ARCH="x86_64"
        BITS=64
        ;;
    *)
        echo -e "${GREEN}*** Unsupported build system ***${NC}"
        echo -e "${GREEN}This script must be run in the MINGW32 or MINGW64 environment${NC}"
        echo -e "${GREEN}that comes with MSYS2.${NC}"
        exit 2
        ;;
esac
if ! test \( -d .git -a -f mypaint.py -a -d gui -a -d lib \); then
    echo -e "${GREEN}*** Not in a MyPaint repository ***${NC}"
    echo -e "${GREEN}This script must be run from the top-level directory of a ${NC}"
    echo -e "${GREEN}MyPaint git repository clone.${NC}"
    exit 2
fi


# Satisfy build dependencies.
# Make sure the build will use the most recent toolchain.

{
  #Added in bypass for Appveyor builds since the dependencies are already installed.
  if ! test "$SKIP_DEPENDENCIES" = "true"; then
    echo -e "${GREEN}+++ Installing toolchain and build deps into your $MSYSTEM ...${NC}"
    pacman -Sy
    pacman -S --noconfirm --needed \
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
  else
    echo -e "${GREEN}+++ Skipping installation of dependencies for Appveyor. Already installed${NC}"
  fi
    echo -e "${GREEN}+++ Installing other required tools...${NC}"
    pacman -S --noconfirm --needed \
        zip \
        tar \
        xz
}


# Determine a source tarball to use.
# This may require invocation of the release script now.
# If exporting from git, the tarball is stored in
# a location which will allow it to be shared between builds.

OUTPUT_ROOT="/tmp/mypaint-builds"

if ! test "x$RELEASE_TARBALL" = "x"; then
    echo -e "${GREEN}+++ Using existing ${RELEASE_TARBALL}.${NC}"
    tarball="$RELEASE_TARBALL"
    EXPORT_ID=`basename "$RELEASE_TARBALL" .tar.xz`
else
    EXPORT_ID=`git rev-parse --short HEAD`
    echo -e "${GREEN}+++ Exporting source from git at ${EXPORT_ID}...${NC}"
    tmp_root="${OUTPUT_ROOT}/${EXPORT_ID}/tmp"
    tarball="$tmp_root"/mypaint.tar.xz
    if ! test -f "$tarball"; then
        mkdir -p "$tmp_root"
        # No git checks because permission semantics differences
        # on permission bits we're using to turn off tests (ugh!)
        # mean spurious `git diff` output on Windows.
        release_opts="--simple-naming --headless --no-gitcheck"
        if ! $RIGOUROUS; then
            release_opts="--no-tests $release_opts"
        fi
        ./release.sh $release_opts -- "$tmp_root"
    fi
fi
if ! test -f "$tarball"; then
    echo -e "${GREEN}*** Tarball $tarball is not available.${NC}"
    exit 2
fi


# Unpack pristine source into an arch-specific src dir
# where the build will take place.

BUILD_ROOT="${OUTPUT_ROOT}/${EXPORT_ID}/${ARCH}"
SRC_DIR="${BUILD_ROOT}/src"

{
    if $RIGOUROUS || ! test -d "$SRC_DIR"; then
        rm -fr "$SRC_DIR"
        mkdir -p "$SRC_DIR"
        tar x -C "$SRC_DIR" --strip-components=1 -pf "$tarball"
    fi
    . "$SRC_DIR/release_info"
    if test "x$MYPAINT_VERSION_FORMAL" = "x"; then
        echo -e "${GREEN}*** $SRC_DIR/release_info did not define MYPAINT_VERSON_FORMAL.${NC}"
        exit 2
    fi
    echo -e "${GREEN}+++ Exported $MYPAINT_VERSION_FORMAL.${NC}"
}


# Begin making a standalone target root folder which will contain a
# mingwXX prefix area as a subdirectory. This will be zipped up as the
# standalone distribution, and later forms the source folder for the
# installer distribution

DIST_BASENAME="mypaint-w${BITS}-${MYPAINT_VERSION_FORMAL}"
TARGET_DIR="${BUILD_ROOT}/${DIST_BASENAME}"
PREFIX="${TARGET_DIR}/mingw${BITS}"

{
    echo -e "${GREEN}+++ Installing runtime dependencies into target...${NC}"
    if $RIGOUROUS; then
        rm -fr "$TARGET_DIR"
    fi
    mkdir -vp "$TARGET_DIR"
    # Set up pacman so it can deploy stuff
    mkdir -vp "$TARGET_DIR/var/lib/pacman"
    mkdir -vp "$TARGET_DIR/var/log"
    mkdir -vp "$TARGET_DIR/tmp"
    pacman -Sy --root "$TARGET_DIR"
    # Alias to simplify things
    pacman_s="pacman -S --root $TARGET_DIR --needed --noconfirm"
    # Need all of numpy
    $pacman_s "${PKG_PREFIX}-python2" \
        "${PKG_PREFIX}-python2-numpy"
    # Avoid a big Perl dep by just installing the icon files
    $pacman_s --assume-installed ${PKG_PREFIX}-icon-naming-utils \
              --ignoregroup ${PKG_PREFIX}-toolchain \
        "${PKG_PREFIX}-adwaita-icon-theme"
    # A subset of the "toolchain" group: runtime stuff only, ideally
    $pacman_s \
        "${PKG_PREFIX}-gcc-libs"
    # Things that depend on the "toolchain" group subset above
    # but which may try to pull in more (avoid that)
    $pacman_s --ignoregroup ${PKG_PREFIX}-toolchain \
              --assume-installed ${PKG_PREFIX}-python3 \
        "${PKG_PREFIX}-json-c" \
        "${PKG_PREFIX}-lcms2" \
        "${PKG_PREFIX}-python2-cairo" \
        "${PKG_PREFIX}-python2-gobject" \
        "${PKG_PREFIX}-librsvg" \
        "${PKG_PREFIX}-gtk3"
    # GSettings runtime requirements
    $pacman_s "${PKG_PREFIX}-gsettings-desktop-schemas"
}


# Handle extra package bundles the maintainer has asked to be installed.
{
    if test -d "$EXTRA_PACKAGES_DIR"; then
        echo -e "${GREEN}+++ Installing extra packages into target...${NC}"
        pacman -U --root $TARGET_DIR --noconfirm \
            "$EXTRA_PACKAGES_DIR"/*.pkg.tar.xz
    else
        echo -e "${GREEN}+++ No extra packages dir: no extras to install.${NC}"
    fi
}


# Install the build of MyPaint
{
    echo -e "${GREEN}+++ Installing MyPaint into the standalone target...${NC}"
    (cd $SRC_DIR && python setup.py build)
    (cd $SRC_DIR && python setup.py install --prefix="" --root="${PREFIX}")
    # Launcher scripts
    cp -v "windows/mypaint-standalone.cmd.in" "$TARGET_DIR/mypaint.cmd"
    sed -i "s|@BITS@|$BITS|g" "$TARGET_DIR/mypaint.cmd"
    cp -v "windows/mypaint-debug.bat" "$PREFIX/bin/"
    # Icons
    cp -v "desktop/mypaint.ico" "$TARGET_DIR/"
    mkdir -p "$PREFIX/share/mypaint"
    cp -v "desktop/mypaint.ico" "$PREFIX/share/mypaint/"
    # Licenses
    mkdir -p "$PREFIX/share/licenses/mypaint"
    cp -v "COPYING" "$PREFIX/share/licenses/mypaint"
    cp -v "Licenses."* "$PREFIX/share/licenses/mypaint"
    mkdir -p "$PREFIX/share/licenses/libmypaint"
}


# Clean up the target - pacman will have pulled in way too much
{
    echo -e "${GREEN}+++ Pruning unnecessary files and folders from the target...${NC}"
    echo -ne "${GREEN}Install size before pruning: ${NC}"
    du -sh "$TARGET_DIR"

    # Docs for non-user-facing things
    rm -fr "$PREFIX"/share/man
    rm -fr "$PREFIX"/share/gtk-doc
    rm -fr "$PREFIX"/share/doc
    rm -fr "$PREFIX"/share/info

    # TCL/tk
    rm -fr "$PREFIX"/lib/tk*
    rm -fr "$PREFIX"/lib/tcl*
    rm -fr "$PREFIX"/lib/itcl*

    # Stuff that's really part of a build environment
    rm -fr "$PREFIX"/share/include
    rm -fr "$PREFIX"/include
    find "$PREFIX"/lib -type f -iname '*.a' -exec rm -f {} \;
    rm -fr "$PREFIX"/lib/python2.7/test
    rm -fr "$PREFIX"/lib/cmake
    rm -fr "$PREFIX"/share/aclocal
    rm -fr "$PREFIX"/share/pkgconfig

    # Remove some FreeDesktop and GNOME things
    rm -fr "$PREFIX"/share/applications
    rm -fr "$PREFIX"/share/appdata
    rm -fr "$PREFIX"/share/mime

    # Other stuff that's not relevant when running on Windows
    rm -fr "$PREFIX"/share/bash-completion
    rm -fr "$PREFIX"/share/readline
    rm -fr "$PREFIX"/share/fontconfig
    rm -fr "$PREFIX"/share/vala
    rm -fr "$PREFIX"/share/thumbnailers   # our fault
    rm -fr "$PREFIX"/share/xml/fontconfig

    # We don't need bitmap icons from the adwaita theme.
    for p in "$PREFIX"/share/icons/Adwaita/*x*; do
        if test -d "$p"; then
            rm -fr "$p"
        fi
    done

    # Strip debugging symbols.
    # Temporarily keeping them in case issue #390 improves. Yeah,
    # unlikely. but.
    #find "$PREFIX" -type f -name "*.exe" -exec strip {} \;
    #find "$PREFIX" -type f -name "*.dll" -exec strip {} \;
    #find "$PREFIX" -type f -name "*.pyd" -exec strip {} \;

    # .pyc bytecode is unnecessary when running with -OO as we do.
    # find "$PREFIX" -type f -name '*.pyc' -exec rm -f {} \;

    # Terminfo is a fairly big db, and we don't need it for MyPaint.
    rm -fr "$PREFIX"/share/terminfo
    rm -fr "$PREFIX"/lib/terminfo

    # Random binaries which aren't going to be invoked. Hopefully.
    find "$PREFIX"/bin -type f \
        -not -iname '*.dll' \
        -not -iname 'python2w.exe' \
        -not -iname 'python2.exe' \
        -not -iname 'gdk-pixbuf-query-loaders.exe' \
        -not -iname 'glib-compile-schemas.exe' \
        -not -iname 'mypaint*' \
        -exec rm -f {} \;

    rm -fr "$TARGET_DIR"/tmp
    if $RIGOUROUS; then
        rm -fr "$TARGET_DIR"/var
    fi
    echo -ne "${GREEN}Install size after pruning: ${NC}"
    du -sh "$TARGET_DIR"
}


# Create a standalone zip bundle while the tree is still pristine

{
    echo -e "${GREEN}+++ Writing standalone zipfile...${NC}"
    rm -f "${OUTPUT_ROOT}/${DIST_BASENAME}.zip"
    rm -f "${BUILD_ROOT}/${DIST_BASENAME}.zip"
    (cd ${BUILD_ROOT} && zip -qXr "${DIST_BASENAME}".zip "$DIST_BASENAME")
    mv -v "${BUILD_ROOT}/${DIST_BASENAME}.zip" "$OUTPUT_ROOT"
    echo -e "${GREEN}+++ Created ${DIST_BASENAME}.zip.${NC}"
}


# Create an Inno Setup compiler script, and start it

{
    echo -e "${GREEN}+++ Making Inno Setup script...${NC}"
    cp -v "windows/innosetup/postinst.cmd" "$PREFIX/bin/mypaint-postinst.cmd"
    cp -v "windows/innosetup/prerm.cmd" "$PREFIX/bin/mypaint-prerm.cmd"
    cp -v "windows/innosetup/wizardimage.bmp" "$TARGET_DIR"
    iss_in="windows/innosetup/mypaint.iss.in"
    iss="$TARGET_DIR/mypaint.iss"
    cp -v "$iss_in" "$iss"
    sed -i "s|@VERSION@|$MYPAINT_VERSION_FORMAL|g" "$iss"
    sed -i "s|@BITS@|$BITS|g" "$iss"
    sed -i "s|@OUTPUTBASENAME@|${DIST_BASENAME}-setup|g" "$iss"
}

{
    echo -e "${GREEN}+++ Running the Inno Setup ISCC tool to make a setup.exe...${NC}"
    rm -f "${BUILD_ROOT}/${DIST_BASENAME}-setup.exe"
    rm -f "${OUTPUT_ROOT}/${DIST_BASENAME}-setup.exe"
    PATH="/c/Program Files (x86)/Inno Setup 5:$PATH"
    if ( cd "$TARGET_DIR" && exec ISCC.exe mypaint.iss ); then
        echo "+++ ISCC ran successfully"
        mv -v "${BUILD_ROOT}/${DIST_BASENAME}-setup.exe" "$OUTPUT_ROOT"
        echo -e "${GREEN}+++ Created ${DIST_BASENAME}-setup.exe${NC}"
    else
        echo -e "${GREEN}*** ISCC failed, see terminal output for details${NC}"
        echo -e "${GREEN}*** (you may need to add the folder with ISCC.exe to your path)${NC}"
    fi
}


# Show the output folder if requested

{
    echo -e "${GREEN}+++ All done.${NC}"
    echo -e "${GREEN}+++ Output can be found in $OUTPUT_ROOT.${NC}"
    if $SHOW_OUTPUT; then
        echo -e "${GREEN}+++ Opening build output folder (--show-output)...${NC}"
        start "$OUTPUT_ROOT"
    fi
    ls -la $OUTPUT_ROOT
}
