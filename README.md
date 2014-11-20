MyPaint
=======
[![Build Status](https://travis-ci.org/mypaint/mypaint.png?branch=master)](https://travis-ci.org/mypaint/mypaint)

MyPaint is a simple drawing and painting program
that works well with Wacom-style graphics tablets.
Its main features are a highly configurable brush engine, speed,
and a fullscreen mode which allows artists to
fully immerse themselves in their work.

* Website: [mypaint.info](http://mypaint.info/)
* Twitter: [@MyPaintApp](https://twitter.com/MyPaintApp)
* Github:
  - [Development "master" branch](https://github.com/mypaint/mypaint)
  - [New issue tracker](https://github.com/mypaint/mypaint/issues)
* Other resources:
  - [Mailing list](https://mail.gna.org/listinfo/mypaint-discuss)
  - [Wiki](http://wiki.mypaint.info/)
  - [Forums](http://forum.intilinux.com/)
  - [Old bug tracker](http://gna.org/bugs/?group=mypaint)
    (patience please: we're migrating bugs across)
  - [Introductory docs for developers](http://wiki.mypaint.info/index.php?title=Documentation/ForDevelopers)

MyPaint is written in Python, C++, and C.
It makes use of the GTK toolkit, version 3.x.
The source is maintained using [git](http://www.git-scm.com),
primarily on Github.

Getting started
===============

MyPaint has an associated library,
[libmypaint](https://github.com/mypaint/libmypaint),
which is distributed as a sister project on Github.
If you fetch the application's source with `git`,
this dependency will be fetched automatically
by the commands below as a relative
[submodule](http://www.git-scm.com/book/en/Git-Tools-Submodules).
There are several third-party dependencies too:

- scons (>= 2.1.0)
- pygobject
- gtk3
- python (= 2.7) (OSX: python >= 2.7.4)
- swig
- numpy
- pycairo (>= 1.4)
- libpng
- lcms2
- libjson-c (>= 0.11, but the older "libjson" name at ~0.10 will work too)
- librsvg

Recommended: a pressure sensitive input device (graphic tablet)

Build & Install (Linux)
-----------------------

* **Install dependencies**: if you run Debian GNU/Linux
  or one of its derivatives like Linux Mint or Ubuntu,
  you can fetch the dependencies by running:

  ```sh
  sudo apt-get install g++ python-dev python-numpy \
  libgtk-3-dev python-gi-dev gir1.2-gtk-3.0 python-gi-cairo \
  swig scons gettext libpng12-dev liblcms2-dev libjson0-dev
  ```
  **Note**: Running `sudo apt-get build-dep mypaint` will install
  most (if not all) of the dependencies for you.

* **Fetch the source**: start by cloning the source repository.
  This will create a directory named "mypaint".
  You should only need to do this initial step once.

  ```sh
  git clone https://github.com/mypaint/mypaint.git
  ```

* **Update submodules**: change into your cloned repository folder,
  and then update the "brushlib" submodule
  so that it contains _libmypaint_ at the correct version:

  ```sh
  cd mypaint
  git submodule update --init --force
  ```

* **Build & test**: starting from your cloned repository folder,
  run _scons_ to compile the C++ and C parts.

  ```sh
  scons
  ```

* **Testing (interactive)**: if the build was successful,
  run the generated script with a clean temporary configuration area
  in order to test that the program works.

  ```sh
  rm -fr /tmp/mypaint_testconfig
  ./mypaint -c /tmp/mypaint_testconfig
  ```

* **Unit tests**: These are purely optional for most users,
  but they're useful for developers and people reporting bugs.
  Please run the unit tests before committing new code,
  and implement doctests for important new Python code.

  ```sh
  sudo apt-get install python-nose
  nosetests --with-doctest
  ```

  - If testing outside a graphical environment (anywhere Gdk refuses
    to initialize), limit the doctests to just `lib/` and `brushlib/`.
  - There are several interactive GUI tests in the `tests/` folder
    which `nosetests` does not run - quite intentionally -
    because their executable bit is set.

* **Updating to the latest source** at a later date is trivial,
  but doing this often means that you have to update the submodule
  or rebuild the compiled parts of the app:

  ```sh
  cd path/to/mypaint
  scons --clean
  git pull
  git submodule update --init --force
  scons
  # ... other commands as necessary ...
  ```

* **To install** MyPaint into the traditional `/usr/local` area
  so that it can be run from your desktop environment:

  ```sh
  cd path/to/mypaint
  sudo scons prefix=/usr/local install
  ```

  - This usually results in entries in menus, launchers, Dashes
    and other desktop environment frippery.

* **To uninstall** the program from a given prefix,
  add the `--clean` option:

  ```sh
  sudo scons prefix=/usr/local install --clean
  ```

Build & Install (Windows)
-------------------------

TBD. [Please help us write this section](https://github.com/mypaint/mypaint/issues/48).

Starting point for up-to-date information:
http://wiki.mypaint.info/Development/Packaging#Windows


Build & Install (Mac)
---------------------

**IN PROGRESS**: [Please help us improve this section](https://github.com/mypaint/mypaint/issues/49).
The wiki's OS X notes are somewhat outdated
and could do with improving too,
but have possibly interesting notes about Quartz vs X11 builds:
http://wiki.mypaint.info/Development/Packaging#OSX. Feedback welcome.

Most users will want to grab `MyPaint-devel`
[from macports](https://www.macports.org/ports.php?by=name&substr=mypaint)
or stick with the stable `MyPaint` portfile already there.

For the adventurous,the following is reported to work on OS X 10.9:

* **Environment setup**: to use Frameworks Python (currently 2.7.8)
  while satisfying the other dependencies from Macports, use

  ```sh
  export PKG_CONFIG_PATH=/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/pkgconfig/
  export CFLAGS="-I/opt/local/include"
  ```

* **Install dependencies**:

  ```sh
  sudo port install gtk3
  sudo port install json-c
  sudo port install py27-numpy
  sudo port install py27-scipy
  sudo port install lcms2
  sudo port install py27-gobject3
  sudo port install hicolor-icon-theme
  ```

* **Fetch source** and **update submodules** just as for Linux:

  ```sh
  git clone https://github.com/mypaint/mypaint.git
  cd mypaint
  git submodule update --init --force
  ```

* **Build and test**.
  The `sudo -E` should not be necessary - let us know if it isn't,
  or if it's unhelpful when running the build in test mode.

  ```sh
  sudo -E scons
  ./mypaint -c /tmp/mypaint_cfgtmp_$$
  ```

Post-install: Linux
-------------------

* **(Advanced) people creating packages** for Linux distributions
  can install as if the prefix were /usr,
  but install the tree somewhere else.
  This can be done as an ordinary user.

  ```sh
  scons prefix=/usr --install-sandbox=`pwd`/path/to/sandbox
  ```

  **NOTE:** the sandbox location must be located under
  the current working directory, and be specified as an *absolute* path.
  You can use ``pwd`` or your build system's absolute
  "path-to-here" variable to achieve that.
  The command above installs the main launch script (for example)
  as `./path/to/sandbox/usr/bin/mypaint`.
  Use a symlink if that's too limiting.

* **(Troubleshooting) runtime linker**: you may need to update
  the runtime linker's caches and links
  after installation on some systems.

  ```sh
  sudo ldconfig
  ```

  Do this if you get any messages about MyPaint
  not being able to load `mypaintlib.so` when run on the command line.

  If you installed to a prefix other than the trusted locations,
  which are typically `/usr/lib` and `/lib`,
  you may need to add a line for your prefix
  into `/etc/ld.so.conf` or `ld.so.conf.d`
  before running `ldconfig`.

  Scons currently won't do this for you because
  the need to perform the action varies by distribution,
  and package distributors need to be able to defer it
  to post-installation scripting.

* **(Troubleshooting) icon theme caches**: take care to update
  the icon theme cache for your prefix
  if you're installing mypaint to a location
  which has one of these files already.
  If you install new icons, any existing icon cache must be updated too,
  otherwise MyPaint won't be able to find its icons
  even if it looks in the right place.

  For example for an install into `/usr`,
  which has an icon cache on most systems,
  you should run:

  ```sh
  sudo gtk-update-icon-cache /usr/share/icons/hicolor
  sudo chmod a+r /usr/share/icons/hicolor/icon-theme.cache
  ```

  after installation to ensure that the cache is up to date.
  Scons currently won't do this for you
  because the cache file is optional.

  If you install to /usr/local, you may need to run this instead:

  ```sh
  gtk-update-icon-cache --ignore-theme-index /usr/local/share/icons/hicolor
  ```

Debugging
=========

By default, our use of Python's ``logging`` module
is noisy about errors, warnings, and general informational stuff,
but silent about anything with a lower priority.
To see all messages, set the ``MYPAINT_DEBUG`` environment variable.

  ```sh
  MYPAINT_DEBUG=1 ./mypaint -c /tmp/cfgtmp_throwaway_1
  ```

MyPaint normally logs Python exception backtraces to the terminal
and to a dialog within the application.

To debug segfaults in C/C++ code, use ``gdb`` with a debug build,
after first making sure you have debugging symbols for Python and GTK3.

  ```sh
  sudo apt-get install gdb python2.7-dbg libgtk-3-0-dbg
  scons debug=1
  export MYPAINT_DEBUG=1
  gdb -ex r --args python ./mypaint -c /tmp/cfgtmp_throwaway_2
  ```

Execute ``bt`` within the gdb environment for a full backtrace.
See also: https://wiki.python.org/moin/DebuggingWithGdb

Legal info
==========

The licenses for various files are described in the LICENSE file.
Documentation can be found within the program and on the homepage:
http://mypaint.info/

A list of contributors can be found in the about dialog.
