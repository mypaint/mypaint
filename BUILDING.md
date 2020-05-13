# Building MyPaint from Source

This guide is for developers, and testers who want the bleeding edge.
Regular users might not need this guide. Most distributions already have
a stable version of MyPaint.

**Table of Contents**

* [Install libmypaint and mypaint-brushes](#install-libmypaint-and-mypaint-brushes)
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
   - [Installing locally](#building-and-installing-locally)
 * [Updating to the latest source](#updating-to-the-latest-source)

## Install libmypaint and mypaint-brushes

MyPaint depends on its brushstroke rendering library,
[**libmypaint**](https://github.com/mypaint/libmypaint).
<details>
  <summary>Which version of libmypaint should I build against?</summary>

When building the latest master, the rule of thumb is to build against
the latest master of libmypaint.

Stable releases should be built against a
compatible stable releases of libmypaint.

If you need to build a commit from the commit history, use `git log`
after having checked out the commit, and search for libmypaint to infer
which commit of libmypaint you should build against.
This is not always specified explicitly, but should always be inferrable by
cross-referencing the commit log of libmypaint (by date or keyword search).
</details>

MyPaint also depends on the default brush collection
[**mypaint-brushes**](https://github.com/mypaint/mypaint-brushes).
These have to be built from scratch for most systems, see the links
below for details on how to do this.

* [Generic libmypaint build instructions][LIB]
* [Generic mypaint-brushes build instructions][BRUSH]
* [Debian-style package builder for libmypaint][LIBDEB]
* [MyPaint's Ubuntu PPA (__not currently updated__)][PPA]

Windows [MSYS2](http://msys2.org) users have pre-packaged options available
for libmypaint-1.3.0 (newer versions currently have to be built from source):

    pacman -S mingw-w64-i686-libmypaint
    pacman -S mingw-w64-x86_64-libmypaint

> ### Using optimization flags
> MyPaint and libmypaint benefit dramatically from autovectorization and other
> compiler optimizations. You may want to set your CFLAGS before compiling:
>
> `
export CFLAGS='-Ofast -ftree-vectorize -fopt-info-vec-optimized -march=native -mtune=native -funsafe-math-optimizations -funsafe-loop-optimizations'
`
>
> To avoid potential glitches, make sure to compile both libmypaint
> and MyPaint using the same optimization flags.

[LIBDEB]: https://github.com/mypaint/libmypaint.deb
[LIB]: https://github.com/mypaint/libmypaint/blob/master/README.md
[BRUSH]: https://github.com/mypaint/mypaint-brushes/blob/master/README.md
[PPA]: https://launchpad.net/~achadwick/+archive/ubuntu/mypaint-testing

## Install third-party dependencies

MyPaint has several third-party dependencies. They all need to be
installed before you can build it.

- setuptools
- pygobject
- gtk3 (>= 3.12)
- python (>= 2.7.4)
- swig (>= 3)
- numpy
- librsvg2 (and its svg gdk-pixbuf loader)
- pycairo (>= 1.4)

Some dependencies have specific versions for Python 2 and Python 3.
Install the ones for the Python version you will use to build MyPaint.
Apart from the use of disk space, there is usually no harm in installing
both sets.

### Debian and derivatives

For Debian, Mint, or Ubuntu, issue the following commands to install the
external dependencies.

    sudo apt-get install -y \
    git swig gettext g++ gir1.2-gtk-3.0 libgtk-3-dev \
    libpng-dev liblcms2-dev libjson-c-dev python-gi-dev \
    librsvg2-common

    # For python 2
    sudo apt-get install -y \
    python-setuptools python-dev python-numpy python-gi-cairo

    # For python 3
    sudo apt-get install -y \
    python3-setuptools python3-dev python3-numpy python3-gi-cairo

If this doesn't work, try older names for the development packages, such
as `libjson0-dev`, or `libpng12-dev`.

### Red Hat and derivatives

For yum-enabled systems, the following should work. This has been tested
on a minimal CentOS 7.3 install, and Fedora 30.

    sudo yum install -y git swig gettext gcc-c++ libpng-devel lcms2-devel \
    json-c-devel gtk3 gtk3-devel gobject-introspection pygobject3-devel \
    librsvg2

    # For python 2
    sudo yum install -y python-setuptools python-devel numpy

    # For python 3
    sudo yum install -y python3-setuptools python3-devel python3-numpy

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
    sudo port install librsvg

These commands are poorly tested, and may be incomplete.
Please send feedback if they're not working for you.

## Fetch the source

Start by cloning the source from git. This will create a new directory
named `mypaint`. Keep this folder around so you don't have to repeat
this step.

    cd path/to/where/I/develop
    git clone https://github.com/mypaint/mypaint.git

## Migration from SCons

We've moved the build system from SCons for portability reasons,
and things may be a bit rough in comparison. If you have an old
installation managed by SCons, please uninstall it before installing
with `setup.py`.

> The SCons files are no longer present in the source tree
> as of revision `a332f03deebebaad84a4f3d5dedc987895dc5b70`.
> To access them, you can check out an earlier revision.
>
> For example: `git checkout a332f03deebebaad84^`

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

The default install location is `/usr/local`. If you want to install
without sudo, refer to [this section](#building-and-installing-locally)

    # You may need to make data files world-readable if you use "sudo"
    sudo find /usr/local -ipath '*mypaint*' -exec chmod -c a+rX {} \;

    # You can uninstall at any later time
    sudo python setup.py managed_uninstall
    sudo python setup.py managed_uninstall --prefix=/usr

Note that uninstallation doesn't get rid of all the folders that the
managed install created.

### Building and installing locally

If you don't want to install MyPaint system-wide, or don't want to use sudo,
follow these instructions to create a local install.

If you also need to install and configure any
[third-party dependencies](#install-third-party-dependencies)
without using sudo, refer to their respective documentation
for how to do so (for most dependencies, this is **_not recommended_**).

In this section, the shell variable `BASE_DIR` is used to refer
to the path of a directory which will be the base of your install.
You can set it like this (modify if you want to install somewhere else):
```
BASE_DIR=$HOME/.local/
```

If you have compatible versions of **libmypaint** and **mypaint-brushes**
installed and configured, all you have to do is run:
```
python setup.py managed_install --prefix=$BASE_DIR
```
and jump to the [run instructions](#running-the-local-installation).
If not, refer to the rest of this section.

#### Installing libmypaint & mypaint-brushes locally

You don't need to install libmypaint or mypaint-brushes **_locally_**
in order to install MyPaint locally, but if you want to, use the
`--prefix=` option to `configure` before running `make install`
for each of them.

E.g:
```
./configure --prefix=$BASE_DIR && make install
```

Refer to [libmypaint's build instructions][LIB]
for more details on building libmypaint.

#### Configuring, building and installing

If you want to use locally installed versions of **libmypaint**
and **mypaint-brushes** you will need to make sure that pkg-config
knows where to find them. To do this, set ```PKG_CONFIG_PATH``` before
building. Assuming both  **libmypaint** and **mypaint-brushes** were
installed configured with ```--prefix=$BASE_DIR``` you can do this by running:

```
export PKG_CONFIG_PATH=$BASE_DIR/lib/pkgconfig/:$BASE_DIR/share/pkgconfig/
```

> The two colon-separated paths refer to the locations of package configuration
> files for libmypaint and mypaint-brushes respectively. Replace the respective
> occurrence of $BASE_DIR if you installed either somewhere else.

In addition to knowing where libmypaint is installed _when building_,
MyPaint also needs to know its location _when running_. This _can_ be
done by setting the `LD_LIBRARY_PATH` environment variable to to the
location of libmypaint every time MyPaint is run, but this is _not_
recommended. The recommended way is to explicitly run the `build_ext`
command with the `--set-rpath` flag, prior to installation.

In short, you can build and install by running:

```
export PKG_CONFIG_PATH=$BASE_DIR/lib/pkgconfig/:$BASE_DIR/share/pkgconfig/
python setup.py build_ext --set-rpath managed_install --prefix=$BASE_DIR
```

> **Note**: remember to use the same prefix if uninstalling via `managed_uninstall`

If you have already run the build script without `--set-rpath`,
you can run the following to force a rebuild:
```
python setup.py build_ext --set-rpath --force
```

> **alternative to `--set-rpath`**
>
> If you want to build an older version of MyPaint that did not have this
> option, you can instead use the built-in `--rpath=` option to `build_ext`,
> setting the dependency path(s) manually.
>
> E.g: `python setup.py build_ext --rpath=$BASE_DIR/lib/`

#### Running the local installation

The start script `mypaint` will be placed in `$BASE_DIR/bin/`, so either add
that path to your PATH environment variable:

```
export PATH=$BASE_DIR/bin/:$PATH
mypaint
```

or create links to the script:

```
ln -s $BASE_DIR/bin/mypaint my-local-mypaint
./my-local-mypaint
```

## Updating to the latest source

Updating to the latest source at a later date is trivial, but doing this
often means that you have to rebuild the compiled parts of the app:

    cd path/to/mypaint
    python setup.py clean --all
    git pull
