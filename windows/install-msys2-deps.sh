#!/bin/sh
# Installs runtime dependencies.
# User-oriented convenience script for MSYS2 installations.
# This shouldn't be used as part of anybody's build scripting.

case "$MSYSTEM" in
    MINGW64)
        ARCH=x86_64
        ;;
    MINGW32)
        ARCH=i686
        ;;
    *)
        echo >&2 "ERROR: unsupported system."
        echo >&2 "This script must be run from either the MINGW32 or the MINGW64 "
        echo >&2 "environment of an MSYS2 installation on Windows."
        exit 2
        ;;
esac

# Update packages caches
pacman -Sy --noconfirm

# Build and runtime deps, all in one go
pacman -S --needed --noconfirm \
    base-devel \
    git \
    mingw-w64-$ARCH-toolchain \
    mingw-w64-$ARCH-pkg-config \
    mingw-w64-$ARCH-gtk3 \
    mingw-w64-$ARCH-json-c \
    mingw-w64-$ARCH-lcms2 \
    mingw-w64-$ARCH-python2-cairo \
    mingw-w64-$ARCH-pygobject-devel \
    mingw-w64-$ARCH-python2-gobject \
    mingw-w64-$ARCH-python2-numpy \
    mingw-w64-$ARCH-hicolor-icon-theme \
    mingw-w64-$ARCH-librsvg

# And for loaders.cache:
pacman -S --noconfirm mingw-w64-$ARCH-gdk-pixbuf2
