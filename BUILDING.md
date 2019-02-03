# Building MyPaint from Source

This guide is for developers, and testers who want the bleeding edge.
Regular users might not need this guide. Most distributions already have
a stable version of MyPaint.

**Table of Contents**

* [Install libmypaint dependency](#install-libmypaint-dependency)
* [Install third-party dependencies](#install-third-party-dependencies)
  - [Debian and derivatives](#debian-and-derivatives)
  - [Red Hat and derivatives](#red-hat-and-derivatives)
  - [Windows MSYS2](#windows-msys2)
  - [OSX MacPorts](#osx-macports)
 * [Fetch the source](#fetch-the-source)
 * [Migration from SCons](#migration-from-scons)
 * [Running the build script](#running-the-build-script)
   - [Demo mode](#demo-mode)
   - [Managed install and uninstall](#managed-install-and-uninstall)
 * [Updating to the latest source](#updating-to-the-latest-source)

## Install libmypaint dependency

MyPaint depends on its brushstroke rendering library,
[libmypaint](https://github.com/mypaint/libmypaint),
at version 2.0.0-alpha or later, as well as [mypaint-brushes](https://github.com/mypaint/mypaint-brushes)
This has to be built from scratch for most systems.

MyPaint and libmypaint benefit dramatically from autovectorization and other compiler optimizations.
You may want to set your CFLAGS before compiling (for gcc):

    $ export CFLAGS='-Ofast -ftree-vectorize -fopt-info-vec-optimized -march=native -mtune=native -funsafe-math-optimizations -funsafe-loop-optimizations'

* [Debian-style package builder for libmypaint][LIBDEB]
* [Generic libmypaint build instructions][LIB]
* [MyPaint's Ubuntu PPA][PPA]

Windows [MSYS2](http://msys2.org) users have pre-packaged options
available:

    pacman -S mingw-w64-i686-libmypaint
    pacman -S mingw-w64-x86_64-libmypaint

[LIBDEB]: https://github.com/mypaint/libmypaint.deb
[LIB]: https://github.com/mypaint/libmypaint/blob/master/README.md
[PPA]: https://launchpad.net/~achadwick/+archive/ubuntu/mypaint-testing

## Install third-party dependencies

MyPaint has several third-party dependencies. They all need to be
installed before you can build it.

- setuptools
- pygobject
- gtk3 (>= 3.12)
- python (>= 2.7.4)
- swig
- numpy
- pycairo (>= 1.4)

### Debian and derivatives

For Debian, Mint, or Ubuntu, issue the following commands to install the
external dependencies.

    sudo apt-get install -y git swig python-setuptools gettext g++
    sudo apt-get install -y python-dev python-numpy
    sudo apt-get install -y libgtk-3-dev python-gi-dev
    sudo apt-get install -y libpng-dev liblcms2-dev libjson-c-dev
    sudo apt-get install -y gir1.2-gtk-3.0 python-gi-cairo

If this doesn't work, try older names for the development packages, such
as `libjson0-dev`, or `libpng12-dev`.

### Red Hat and derivatives

For yum-enabled systems, the following should work. This has been tested
on a minimal CentOS 7.3 install.

    sudo yum install -y git swig python-setuptools gettext gcc-c++
    sudo yum install -y python-devel numpy
    sudo yum install -y gtk3-devel pygobject3-devel
    sudo yum install -y libpng-devel lcms2-devel json-c-devel
    sudo yum install -y gtk3 gobject-introspection

### Windows MSYS2

Use the following commands when building in [MSYS2](http://msys2.org).
For 32-bit targets, use "i686" in place of the "x86_64".

    pacman -S --noconfirm --needed git base-devel
    pacman -S --noconfirm --needed  \
      mingw-w64-x86_64-toolchain     \
      mingw-w64-x86_64-pkg-config     \
      mingw-w64-x86_64-python2-numpy   \
      mingw-w64-x86_64-gtk3            \
      mingw-w64-x86_64-pygobject-devel \
      mingw-w64-x86_64-lcms2           \
      mingw-w64-x86_64-json-c           \
      mingw-w64-x86_64-librsvg           \
      mingw-w64-x86_64-hicolor-icon-theme \
      mingw-w64-x86_64-python2-cairo      \
      mingw-w64-x86_64-python2-gobject    \
      mingw-w64-x86_64-mypaint-brushes2

### OSX MacPorts

To use Frameworks Python (currently 2.7.8) while satisfying the other
dependencies from Macports, use:

    export PKG_CONFIG_PATH=/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/pkgconfig/
    export CFLAGS="-I/opt/local/include"

    sudo port install gtk3
    sudo port install py27-numpy
    sudo port install py27-scipy
    sudo port install py27-pyobjc-cocoa    # optional, for i18n support
    sudo port install py27-gobject3
    sudo port install json-c
    sudo port install lcms2
    sudo port install hicolor-icon-theme

These commands are poorly tested, and may be incomplete.
Please send feedback if they're not working for you.

## Fetch the source

Start by cloning the source from git. This will create a new directory
named `mypaint`. Keep this folder around so you don't have to repeat
this step.

    cd path/to/where/I/develop
    git clone https://github.com/mypaint/mypaint.git

## Migration from SCons

We've just moved the build system from SCons for portability reasons,
and things may be a bit rough in comparison. If you have an old
installation managed by SCons, please uninstall it before installing
with `setup.py`.

Real Pythonistas™ might expect `pip` to work. It doesn't, not yet:
MyPaint has way too many support files that have to be in special
folders, so it uses a custom installation scheme.

## Running the build script

MyPaint is a Python project, and it uses a conventional `setup.py`
script. However, this isn't a typical Python module, so `pip install`
doesn't work with it yet.

    # Learn how to run setup.py
    cd mypaint
    python setup.py --help-commands   # list all commands
    python setup.py --help build   # get options for "build"

    # Some basic commands
    python setup.py build
	python setup.py clean --all

We've added a few custom commands too, for people used to the old SCons
way of working.

    # Test without a full installation
    python setup.py demo

    # Don't use raw "install" unless you know what you're doing
    python setup.py managed_install
    python setup.py managed_uninstall

See above if you want to install MyPaint or use `pip`. This isn't a
conventional installation scheme.

### Demo mode

You can test MyPaint without installing it. The settings aren't saved
between runs when you do this.

    python setup.py demo

### Unit tests

Please run the doctests before committing new code.

    sudo apt-get install python-nose
    python setup.py nosetests

We have some heavier conformance tests for the C++ parts too. These take
longer to run.

    python setup.py test

You should write doctests for important new Python code. Please consider
writing tests in the `tests` folder too, if you make any changes to the
C++ extension or `libmypaint`.

To cleanup between unit tests you may want to run:

    python setup.py clean --all
    rm -vf lib/*_wrap.c*

### Managed install and uninstall

MyPaint has an additional custom install command, for people used to our
old SCons recipes. It isn't compatible with SCons installs, but it
allows you to do an uninstall later.

    # For most Linux types
    sudo python setup.py managed_install
    sudo python setup.py managed_install --prefix=/usr

The default install location is `/usr/local`.

    # You may need to make data files world-readable if you use "sudo"
    sudo find /usr/local -ipath '*mypaint*' -exec chmod -c a+rX {} \;

    # You can uninstall at any later time
    sudo python setup.py managed_uninstall
    sudo python setup.py managed_uninstall --prefix=/usr

Note that uninstallation doesn't get rid of all the folders that the
managed install created.

## Updating to the latest source

Updating to the latest source at a later date is trivial, but doing this
often means that you have to rebuild the compiled parts of the app:

    cd path/to/mypaint
    python setup.py clean --all
    git pull
