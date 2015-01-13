Building MyPaint on Windows
===========================

NOTE: THIS STYLE OF BUILD IS NOT YET READY FOR GENERAL DISTRIBUTION

This document describes building a native Win32 MyPaint using MSYS2's
MinGW-w64. The resultant script and its extension DLL must be run using
the native Python which built it, and running it depends on external
libs like GTK3 which reside in the native `/mingw32` tree. The only
supported method for running MyPaint with these instructions is from the 

This doc DOES NOT COVER building MyPaint as a single .exe, or into any
form of installer. That is a thing we want to do eventually, and we'd be
happy to accept patches and new build procedures which make it happen;
however this document currently covers only the bare essentials needed
to get MyPaint running for debugging purposes.

In other words, the build created with these instructions compiles, but
the program isn't usable yet. If you're curious however, read on.

Get and update MSYS2
--------------------

MSYS2 is a free toolchain, build environment, and binary library
distribution for Windows. It can create native PE32 binaries for Windows
using MinGW-w64. We'll be building for its "MINGW32" target.

At the time of writing, NumPy on native Win64 is not considered stable
enough for production when compiled with MinGW. It emits several
warnings when used, so we'll give it a miss this time round.

Follow the instructions at either

* http://sourceforge.net/p/msys2/wiki/MSYS2%20installation/ or
* https://msys2.github.io/

to get started. The docs below use `msys2-x86_64-20141113.exe` as their
starting point.

One installed, update MSYS2:

    pacman -Sy
    pacman --needed -S bash pacman msys2-runtime
    ; then close and reopen the shell
    pacman -Su
    ; restart shell again

The commands above can be run in any shell shipping with MSYS2, but the
build commands below must be run in the "MinGW-w64 Win32 Shell".

Developer tools
---------------

Install some platform-independent GNU-ish developer tools:

    pacman -S git automake-wrapper autoconf autogen \
      libtool m4 make swig

For compiling `json-c` (and MyPaint), we'll need a platform-specific
compiler and pkg-config too:

    pacman -S mingw-w64-i686-pkg-config  mingw-w64-i686-gcc

We're *not installing base-devel* here because that pulls in a version
of SCons and the Python it depends on, but for the Cygwin-like MSYS2
environment only. For various messy reasons, this can't be used to build
Python extensions for the native MINGW32 target. If you've somehow ended
up with an MSYS2 Python or SCons, uninstall it now:

    pacman -Rc python2

or you will be entering a sea of frustration.

Install libjson-c
-----------------

libmypaint needs `libjson-c`, which doesn't have a build script yet.
For now, it needs a manual build and install:

    cd /usr/src
    git clone https://github.com/json-c/json-c.git
    cd json-c
    sh autogen.sh
    ./configure --prefix=/mingw32
    make
    make install

You can use anything you like in place of `/usr/src`.

Install other MyPaint dependencies
----------------------------------

Most MyPaint dependencies already have a build script and precompiled
binaries. Kudos to everyone maintaining [MINGW-packages][1] for MSYS2!

    pacman -S mingw-w64-i686-gtk3 \
      mingw-w64-i686-lcms2 \
      mingw-w64-i686-python2-cairo \
      mingw-w64-i686-pygobject-devel \
      mingw-w64-i686-python2-gobject \
      mingw-w64-i686-python2-numpy \
      mingw-w64-i686-hicolor-icon-theme \
      mingw-w64-i686-librsvg

Make sure that GdkPixbuf's `loaders.cache` gets regenerated for `librsvg` so
that MyPaint will be able to load its symbolic icons. The quickest way is to
force a reinstall of the package:

    pacman -S mingw-w64-i686-gdk-pixbuf2

but you can regenerate it with `gdk-pixbuf-query-loaders.exe` too.

Another thing we need is SCons. For our purposes, it's best to run
SCons-local with the native Python2 build which the above will have
installed into `/mingw32`. SCons-local can be downloaded from
http://scons.org/download.php, or just do.

    pacman -S wget
    cd /usr/src
    wget http://prdownloads.sourceforge.net/scons/scons-local-2.3.4.tar.gz

Build and test MyPaint
----------------------

Finally, fetch and build MyPaint itself:

    cd /usr/src
    git clone https://github.com/mypaint/mypaint.git
    cd mypaint
    tar xzf ../scons-local-2.3.4.tar.gz
    git submodule update --init
    MSYSTEM= ./scons.py

Note the need to unset `MSYSTEM` when SCons runs. The Python we'll be
using has [some oddities with path separators][2] which make this
necessary.

Hopefully after this, MyPaint can be run from the location you pulled it
down to:

    ./mypaint -c /tmp/cfgtmp1

At the time of writing however, MyPaint is fairly riddled with
showstopper bugs on Win32. But at least now we can work on those.

[1]: https://github.com/Alexpux/MINGW-packages
[2]: https://github.com/Alexpux/MINGW-packages/blob/94b907b38e569fb00c60b564b14a06fe38101ee4/mingw-w64-python2/0600-msys-mingw-prefer-unix-sep-if-MSYSTEM.patch#L18
