# Building MyPaint on Windows

## Setup.exe and standalone

The `windows` subdirectory contains scripting for making a Win32
installer from git. This may be all you need to build MyPaint from
source on Windows. See [windows/README.md][1] for detailed instructions.

The remainder of this document describes a manual process intended for
developers.

## Manual building and testing

This document describes building a native Win32 MyPaint using MSYS2's
MinGW-w64. The resultant script and its extension DLL must be run with
MSYS2's native Python build. Running it depends on external libs like
GTK3 packaged by the MSYS2 team, all of which also reside in the native
`/mingw32` tree.

The only supported method for running MyPaint with these instructions is
from the command line.

This doc DOES NOT COVER building MyPaint into an installer bundle See
[windows/README.md][1] if you want to do that. This document covers
only the bare essentials needed to get MyPaint running for debugging
purposes.

### Get and update MSYS2

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

    pacman --needed -Sy bash pacman msys2-runtime pacman-mirrors
    ; then close and reopen the shell
    pacman -Syu
    ; restart shell again

The commands above can be run in any shell shipping with MSYS2, but the
build commands below must be run in the "MinGW-w64 Win32 Shell".

### Developer tools

Install the target-independent developer tools needed for the build,
and `git` for fetching the source.
The latter is not needed if you're building a tarball
or have retrieved the MyPaint code in some other way.

    pacman -S base-devel git

For compiling MyPaint, we'll need a target-specific build
toolchain and `pkg-config` utility.
The instructions below assume the `i686` target, i.e. a 32-bit build.
You can substitute `x86_64` if you want a 64-bit build,
but be aware that 64-bit MinGW builds for Windows more experimental.

    pacman -S mingw-w64-i686-toolchain mingw-w64-i686-pkg-config

### Install MyPaint dependencies

All of MyPaint's dependencies are available from the MSYS2 repositories.
Thanks to everyone maintaining [MINGW-packages][2] for giving us
a great open platform to build against!

    pacman -S mingw-w64-i686-gtk3 \
      mingw-w64-i686-json-c \
      mingw-w64-i686-lcms2 \
      mingw-w64-i686-python2-cairo \
      mingw-w64-i686-pygobject-devel \
      mingw-w64-i686-python2-gobject \
      mingw-w64-i686-python2-numpy \
      mingw-w64-i686-hicolor-icon-theme \
      mingw-w64-i686-librsvg

Make sure to regenerate GdkPixbuf's `loaders.cache` for `librsvg`
so that MyPaint will be able to load its symbolic icons.
The quickest way to do this
is to force a reinstall of the package:

    pacman -S mingw-w64-i686-gdk-pixbuf2

but you can regenerate it with `gdk-pixbuf-query-loaders.exe` too.

### Build and test MyPaint

Start by fetching and building MyPaint.
You need to do this from the MINGW32 environment.
Start by running "MinGW-w64 Win32 Shell" from the Start menu.
Do not use the MSYS2 shell for this stage.

    cd /usr/src
    git clone https://github.com/mypaint/mypaint.git
    cd mypaint
    git submodule update --init
    scons

The "scons" used here is actually the MSYS environment's scons.
We have several nasty hacks in our SCons scripting to make this work,
but "building from MSYS2's MINGW32 using MSYS scons" is a platform
combination we specifically support.

Hopefully after this, MyPaint can be run
from the location you pulled it down to:

    ./mypaint -c /tmp/cfgtmp1

MyPaint may be quite a bit more buggy on the Windows platform
than on the Linux platform, be warned.

## Known Problems

* **No pressure support / glitches with tablet drivers.**
  These should be reported to the GDK maintainers.
  See <https://bugzilla.gnome.org/show_bug.cgi?id=743330>
  for an example of how to do this effectively.
  Discussing your issue on [IRC](irc://irc.gnome.org/%23gtk%2B)
  after raising it in the tracker is often very fruitful too.

* **Bugs. Lots of them.**
  The port to Windows has historically received the least love of all
  MyPaint ports, but this document is intended to help address that.
  The number is diminishing, and MyPaint runs on native Windows
  reasonably well now.
  We really need actively testing users to improve support further.
  Please report problems as described in [CONTRIBUTING.md](CONTRIBUTING.md).

[1]: ./windows/README.md
[2]: https://github.com/Alexpux/MINGW-packages
