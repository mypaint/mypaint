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

Install the target-independent developer tools needed for the build, and `git` for fetching the source. The latter is not needed if you're building a tarball or have retreived the MyPaint code in some other way.

    pacman -S base-devel git

For compiling MyPaint, we'll need a target-specific build toolchain and `pkg-config` utility. The instructions below assume the `i686` target, i.e. a 32-bit build. You can substitute `x86_64` if you want a 64-bit build, but be aware that 64-bit MinGW builds for Windows more experimental.

    pacman -S mingw-w64-i686-toolchain mingw-w64-i686-pkg-config

Install MyPaint dependencies
----------------------------

All of MyPaint's dependencies are available from the MSYS2 repositories.
Thanks to everyone maintaining [MINGW-packages][1] for giving us a great
open platform to build against!

    pacman -S mingw-w64-i686-gtk3 \
      mingw-w64-i686-json-c \
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


Build and test MyPaint
----------------------

Unfortunately, base-devel's version of SCons seems ignorant of
MINGW32 tools and prefixes at version 2.3.4-2. We don't have a
workaround which would let us use that scons yet, but the quickest
fix is to use SCons-local with the native build of Python2 which
should already be installed into `/mingw32`. SCons-local can be
downloaded from http://scons.org/download.php, or just do.

    pacman -S wget
    cd /usr/src
    wget http://prdownloads.sourceforge.net/scons/scons-local-2.3.4.tar.gz
    mkdir -p scons-local
    tar xzf scons-local-2.3.4.tar.gz -C scons-local

Once that's done, fetch and build MyPaint itself. You need to do this from
the MINGW32 environment.

    cd /usr/src
    git clone https://github.com/mypaint/mypaint.git
    cd mypaint
    tar xzf ../scons-local-2.3.4.tar.gz
    git submodule update --init
    MSYSTEM= ../scons-local/scons.py

Note the need to unset `MSYSTEM` when SCons runs. The Python we'll be
using has [some oddities with path separators][2] which make this
necessary.

Hopefully after this, MyPaint can be run from the location you pulled it
down to:

    ./mypaint -c /tmp/cfgtmp1

At the time of writing however, MyPaint is fairly riddled with
showstopper bugs on Win32. But at least now we can work on those.

Known Problems
--------------

* **No pressure support / glitches with tablet drivers.**
  One possible cause of this is being actively worked on:
  it seems to be a GDK bug relating to WinTab initialization.
  See https://bugzilla.gnome.org/show_bug.cgi?id=743330

* **Bugs. Huge numbers of them.**
  The port to Windows has historically received the least love of all
  MyPaint ports, but this document is intended to help address that.
  We really need actively testing users to improve matters.
  Please report problems as described in [CONTRIBUTING.md](CONTRIBUTING.md).

* **No standardized pre-packaging.**
  One will be needed before this build can be distributed in any
  form which is meaningful to ordinary people.
  A possible starting point is http://www.scons.org/doc/HTML/scons-man.html#b-Package.

[1]: https://github.com/Alexpux/MINGW-packages
[2]: https://github.com/Alexpux/MINGW-packages/blob/94b907b38e569fb00c60b564b14a06fe38101ee4/mingw-w64-python2/0600-msys-mingw-prefer-unix-sep-if-MSYSTEM.patch#L18
